import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)

import os
import torch
from torch.utils.data import DataLoader

from vocabulary import Vocab, get_blank_index

from src.deepspeech import Model
from src.optimization import training
from src.audio_utils import get_default_audio_transforms, SpectrogramTransform
from src.datasets import AudioDataset, AudioDatasetSampler, AudioDataloaderWrapper
from src.datasets import collate_fn, convert_libri_manifest_to_common_voice


# Set proper device for computations,
dtype, device, cuda_device_id = torch.float32, None, 0
os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(str(cuda_device_id) if cuda_device_id is not None else '')
if cuda_device_id is not None and torch.cuda.is_available():
    device = 'cuda:{0:d}'.format(0)
else:
    device = torch.device('cpu')
    
print(f'dtype: {dtype}, device: {device}, cuda_device_id {cuda_device_id}')


alphabet = [
    'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к',
    'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц',
    'ч', 'ш', 'щ', 'ь', 'ы', 'ъ', 'э', 'ю', 'я', ' ', '<blank>'
]

vocab = Vocab(alphabet)

blank_index = get_blank_index(vocab)

audio_transforms = get_default_audio_transforms()
# audio_transforms = None

sample_rate = 8000

# ## Load Common Voice dataset
common_voice_val_manifest_path = '/home/e.chuykova/data/val.txt'
common_voice_test_manifest_path = '/home/e.chuykova/data/test.txt'
common_voice_train_manifest_path = '/home/e.chuykova/data/train.txt'

common_voice_val_dataset = AudioDataset(
    common_voice_val_manifest_path, vocab, sample_rate=sample_rate,
)
common_voice_test_dataset = AudioDataset(
    common_voice_test_manifest_path, vocab, sample_rate=sample_rate,
)
common_voice_train_dataset = AudioDataset(
    common_voice_train_manifest_path, vocab, sample_rate=sample_rate,
    audio_transforms=audio_transforms,
)

# ## Load LibriSpeech dataset
ls_dev_manifest_path = '/data/mnakhodnov/voice_data/libri_speech/dev/manifest.json'
ls_test_manifest_path = '/data/mnakhodnov/voice_data/libri_speech/test/manifest.json'
ls_train_manifest_path = '/data/mnakhodnov/voice_data/libri_speech/train/manifest.json'

ls_dev_manifest_path = convert_libri_manifest_to_common_voice(ls_dev_manifest_path)
ls_test_manifest_path = convert_libri_manifest_to_common_voice(ls_test_manifest_path)
ls_train_manifest_path = convert_libri_manifest_to_common_voice(ls_train_manifest_path)

ls_dev_dataset = AudioDataset(
    ls_dev_manifest_path, vocab=vocab, sample_rate=sample_rate, max_duration=10.0,
)
ls_test_dataset = AudioDataset(
    ls_test_manifest_path, vocab=vocab, sample_rate=sample_rate, max_duration=10.0,
)
ls_train_dataset = AudioDataset(
    ls_train_manifest_path, vocab=vocab, sample_rate=sample_rate, max_duration=10.0,
    audio_transforms=audio_transforms,
)

# ## Load Open STT (radio_2) dataset

open_stt_manifest_path = '/data/mnakhodnov/voice_data/radio_2/radio_2.common_voice.csv'
open_stt_test_manifest_path = '/data/mnakhodnov/voice_data/radio_2/radio_2.common_voice_test.csv'
open_stt_train_manifest_path = '/data/mnakhodnov/voice_data/radio_2/radio_2.common_voice_train.csv'

open_stt_test_dataset = AudioDataset(
    open_stt_test_manifest_path, vocab=vocab, sample_rate=sample_rate, min_duration=2.0, max_duration=10.0,
)
open_stt_train_dataset = AudioDataset(
    open_stt_train_manifest_path, vocab=vocab, sample_rate=sample_rate, min_duration=2.0, max_duration=10.0,
    audio_transforms=audio_transforms,
)

audiobook_2_test_manifest_path = '/data/mnakhodnov/voice_data/private_buriy_audiobooks_2/private_buriy_audiobooks_2.common_voice_test.csv'
audiobook_2_train_manifest_path = '/data/mnakhodnov/voice_data/private_buriy_audiobooks_2/private_buriy_audiobooks_2.common_voice_train.csv'

audiobook_2_test_dataset = AudioDataset(
    audiobook_2_test_manifest_path, vocab=vocab, sample_rate=sample_rate, min_duration=2.0, max_duration=10.0,
)

# ## Combine all datasets for training
combined_dataset = AudioDataset(
    [common_voice_train_manifest_path, ls_train_manifest_path,
     open_stt_train_manifest_path, audiobook_2_train_manifest_path],
    vocab=vocab, sample_rate=sample_rate,
    min_duration=[None, None, 2.0, 2.0], max_duration=[None, 10.0, 10.0, 10.0],
    audio_transforms=audio_transforms,
)

# # Create Dataloaders
batch_size = 80
train_num_workers, val_num_workers = 8, 4

# ## Common Voice
common_voice_val_dataloader = AudioDataloaderWrapper(DataLoader(
    common_voice_val_dataset, batch_size=batch_size, shuffle=False, 
    num_workers=val_num_workers, pin_memory=False, collate_fn=collate_fn
))
common_voice_test_dataloader = AudioDataloaderWrapper(DataLoader(
    common_voice_test_dataset, batch_size=batch_size, shuffle=False, 
    num_workers=val_num_workers, pin_memory=False, collate_fn=collate_fn
))

# ## Libri Speech
ls_dev_dataloader = AudioDataloaderWrapper(DataLoader(
    ls_dev_dataset, batch_size=batch_size, shuffle=False, 
    num_workers=val_num_workers, pin_memory=False, collate_fn=collate_fn
))
ls_test_dataloader = AudioDataloaderWrapper(DataLoader(
    ls_test_dataset, batch_size=batch_size, shuffle=False, 
    num_workers=val_num_workers, pin_memory=False, collate_fn=collate_fn
))

# ## OpenSTT
open_stt_test_dataloader = AudioDataloaderWrapper(DataLoader(
    open_stt_test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=val_num_workers, pin_memory=False, collate_fn=collate_fn
))

audiobook_2_test_dataloader = AudioDataloaderWrapper(DataLoader(
    audiobook_2_test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=val_num_workers, pin_memory=False, collate_fn=collate_fn
))

# ## Combined Dataloader
combined_dataloader = AudioDataloaderWrapper(DataLoader(
    combined_dataset, batch_size=batch_size, 
    sampler=AudioDatasetSampler(combined_dataset, batch_size=batch_size),
    num_workers=train_num_workers, pin_memory=False, collate_fn=collate_fn
))

# # Create Model
# ## Choose LM for beam search decoder

# This models are sorted wr to their size and speed 
# kenlm_data_path = '/data/mnakhodnov/language_data/cc100/xaa.processed.1'
# kenlm_data_path = '/data/mnakhodnov/language_data/cc100/xaa.processed.2'
# kenlm_data_path = '/data/mnakhodnov/language_data/cc100/xaa.processed.3'
# kenlm_data_path = '/data/mnakhodnov/language_data/cc100/xaa.processed.4'
kenlm_data_path = '/data/mnakhodnov/language_data/common_voice/train.txt'
kenlm_arpa_path, kenlm_binary_path = kenlm_data_path + '.arpa', kenlm_data_path + '.binary'

fast_beam_kwargs = {
    'beam_size': 10, 'cutoff_top_n': 5, 'cutoff_prob': 1.0, 
    'ext_scoring_func': kenlm_binary_path, 'alpha': 1.0, 'beta': 0.3, 'num_processes': 32
}


def load_from_ckpt(_model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    _model.load_state_dict(checkpoint['model_state_dict'])


num_tokens = len(vocab.tokens2indices()) - 1
num_mel_bins = 64
hidden_size = 512
num_layers = 4

model_dir = 'models/9'
log_every_n_batch = 10

model = Model(
    num_mel_bins=num_mel_bins,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_tokens=num_tokens
)
load_from_ckpt(model, '/home/mnakhodnov/sirius-stt/models/8_recovered_v3/epoch_5.pt')
model = model.to(device=device)

from torch.optim.lr_scheduler import ExponentialLR

learning_rate = 2e-4
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CTCLoss(blank=blank_index, reduction='mean', zero_infinity=True)

scheduler = ExponentialLR(opt, gamma=0.9)
# spectrogram_transform = None
# spectrogram_transform_first_epoch = None

# spectrogram_transform = SpectrogramTransform(freq_mask_param=10, time_mask_param=10)
# spectrogram_transform_first_epoch = None

spectrogram_transform = SpectrogramTransform(freq_mask_param=15, time_mask_param=15)
spectrogram_transform_first_epoch = 0

num_epochs = 100

grad_scaler = torch.cuda.amp.GradScaler()
training(
    model=model, optimizer=opt, loss_fn=loss_fn, num_epochs=num_epochs, 
    train_dataloader=[combined_dataloader, 'combined/train'],
    val_dataloaders={
        'audiobook_2/test': audiobook_2_test_dataloader,
        'open_stt/test': open_stt_test_dataloader,
        'libre_speech/dev': ls_dev_dataloader,
        'libre_speech/test': ls_test_dataloader,
        'common_voice/val': common_voice_val_dataloader,
        'common_voice/test': common_voice_test_dataloader,
    }, log_every_n_batch=log_every_n_batch, model_dir=model_dir, vocab=vocab,
    beam_kwargs=fast_beam_kwargs, 
    spectrogram_transform=spectrogram_transform, 
    spectrogram_transform_first_epoch=spectrogram_transform_first_epoch,
    scheduler=scheduler, grad_scaler=grad_scaler
)
