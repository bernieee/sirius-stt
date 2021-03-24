import gc
import os
import time
import torch
from typing import Optional

import tqdm.autonotebook as tqdm

from src.logging_my import TensorboardLogger
from src.audio_utils import compute_log_mel_spectrogram
from src.decoding import greedy_decoder, calc_wer_for_batch, fast_beam_search_decode


def get_prediction(logprobs, logprobs_lens, vocab, decoder=greedy_decoder, decoder_kwargs=None):
    """ Compute greedy text from loglikes.
            Input shape:
                logprobs: 3D tensor with shape (num_timesteps, batch_size, alphabet_len)
                logprobs_lens: 1D tensor with shape (batch_size)
            Returns:
                list of texts with len (batch_size)
    """
    if decoder_kwargs is None:
        decoder_kwargs = dict()

    predictions = decoder(logprobs=logprobs, logprobs_lens=logprobs_lens, vocab=vocab, **decoder_kwargs)
    predictions = [sorted(hypos, key=lambda key_value: key_value[1], reverse=True)[0][0] for hypos in predictions]

    return predictions


def get_model_results(
        model, audios, audio_lens, tokens, texts, text_lens,
        vocab, loss_fn, decoder, decoder_kwargs, spectrogram_transform=None
):
    """ get mean loss, mean wer and prediction list for batch
        Returns:
            loss: int
            wer: int
            prediction: list of str

    """
    # write your code here
    log_mel_spectrogram, seq_lens = compute_log_mel_spectrogram(
        audios, audio_lens, spectrogram_transform=spectrogram_transform
    )
    logprobs, seq_lens = model(log_mel_spectrogram, seq_lens)
    loss = loss_fn(logprobs, tokens, seq_lens, text_lens)

    with torch.no_grad():
        predictions = get_prediction(logprobs, seq_lens, vocab, decoder=decoder, decoder_kwargs=decoder_kwargs)
        wer = calc_wer_for_batch(predictions, texts)
    return loss, wer, predictions


def validate(model, dataloader, vocab, loss_fn, decoder, decoder_kwargs):
    device = next(iter(model.parameters())).device

    loss, wer = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
            batch = {
                key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in
                batch.items()
            }
            # Note: Fix for dataleak
            batch['texts'] = list(map(lambda text: text.decode('utf-8'), batch['texts']))

            loss_batch, wer_batch, prediction = get_model_results(
                model, batch["audios"], batch["audio_lens"],
                batch["tokens"], batch["texts"], batch["text_lens"], vocab, loss_fn,
                decoder=decoder, decoder_kwargs=decoder_kwargs, spectrogram_transform=None
            )

            loss += loss_batch.item() * batch['audio_lens'].shape[0]
            wer += wer_batch * batch['audio_lens'].shape[0]

        loss /= len(dataloader.dataset)
        wer /= len(dataloader.dataset)
    return loss, wer, prediction, batch["texts"]


# noinspection PyProtectedMember
def training(
        model, optimizer, loss_fn, num_epochs,
        train_dataloader, val_dataloaders, log_every_n_batch, model_dir,
        vocab, beam_kwargs, spectrogram_transform=None, spectrogram_transform_first_epoch=None,
        scheduler: Optional[torch.optim.lr_scheduler.ExponentialLR] = None
):
    device = next(iter(model.parameters())).device

    logger = TensorboardLogger(model_dir)

    if isinstance(train_dataloader, list):
        train_dataloader, train_dataloader_name = train_dataloader
    else:
        train_dataloader, train_dataloader_name = train_dataloader, 'train'

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_wer = 0, 0
        model.train(True)

        using_spectrogram_transform = spectrogram_transform
        if spectrogram_transform is not None:
            if spectrogram_transform_first_epoch is not None and epoch < spectrogram_transform_first_epoch:
                using_spectrogram_transform = None

        logger.writer.add_scalar(f'lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        for iteration, batch in enumerate(tqdm.tqdm(train_dataloader, total=len(train_dataloader))):
            batch = {
                key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in
                batch.items()
            }
            # Note: Fix for dataleak
            batch['texts'] = list(map(lambda text: text.decode('utf-8'), batch['texts']))

            loss, wer, prediction = get_model_results(
                model, batch["audios"], batch["audio_lens"],
                batch["tokens"], batch["texts"], batch["text_lens"], vocab, loss_fn,
                decoder=greedy_decoder, decoder_kwargs=dict(), spectrogram_transform=using_spectrogram_transform
            )

            # optimizer step
            # write your code here
            optimizer.zero_grad()
            loss.backward()

            gradient_norm = torch.sqrt(sum((torch.square(torch.norm(p.grad)) for p in model.parameters()))).item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0, norm_type=2.0)
            gradient_norm_clipped = torch.sqrt(
                sum((torch.square(torch.norm(p.grad)) for p in model.parameters()))).item()

            optimizer.step()

            train_loss += loss.item() * batch['audio_lens'].shape[0]
            train_wer += wer * batch['audio_lens'].shape[0]

            step = len(train_dataloader) * epoch + iteration

            logger.writer.add_scalar(f'grad_norm/non_clipped', gradient_norm, global_step=step)
            logger.writer.add_scalar(f'grad_norm/clipped', gradient_norm_clipped, global_step=step)
            if step % log_every_n_batch == 0:
                logger.log(step, loss, wer, train_dataloader_name)
                logger.log_text(step, prediction, batch["texts"], train_dataloader_name)
                gc.collect()

            del batch, loss, wer, prediction
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, os.path.join(model_dir, f'epoch_{epoch}.pt'))

        if scheduler is not None:
            scheduler.step()

        train_loss /= len(train_dataloader.dataset)
        train_wer /= len(train_dataloader.dataset)

        model.train(False)
        val_losses, val_wers = dict(), dict()
        for name, val_dataloader in val_dataloaders.items():
            val_loss, val_wer, prediction, batch_texts = validate(
                model=model, dataloader=val_dataloader, vocab=vocab, loss_fn=loss_fn,
                decoder=fast_beam_search_decode, decoder_kwargs=beam_kwargs
            )
            val_wers[name] = val_wer
            val_losses[name] = val_loss

            logger.log(epoch, val_loss, val_wer, f'{name}')
            logger.log_text(epoch, prediction, batch_texts, f'{name}')

        common_voice_val_wer, common_voice_val_loss = val_wers['common_voice/val'], val_losses['common_voice/val']
        print(
            f'\nEpoch {epoch + 1} of {num_epochs} took {time.time() - start_time}s, '
            f'train loss: {train_loss}, val loss: {common_voice_val_loss}, '
            f'train wer: {train_wer}, val wer: {common_voice_val_wer}'
        )

    logger.close()
    print("Finished!")
