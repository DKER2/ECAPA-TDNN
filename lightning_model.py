import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning import LightningModule, Trainer, seed_everything
from loss import AAMsoftmax, Phoneme_SSL_loss
from model import ECAPA_TDNN

class Task(LightningModule):
    def __init__( self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
        super(self).__init__()
        ## ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C = C)

        ## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s)
		self.phoneme_loss    = Phoneme_SSL_loss(num_frames=20, num_sample=3)

        self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
    
    def training_step(self, batch, batch_idx):
        data, seq_len, labels = batch
        labels = torch.LongTensor(labels)
		speaker_embedding, phonemes, seq_len = self.speaker_encoder.forward(data, aug = True)
		nloss, prec  = self.speaker_loss.forward(speaker_embedding, labels)
		loss_phn = self.phoneme_loss.forward(phonemes, seq_len)
		loss = loss_phn
        self.log('train_loss', loss, prog_bar=True)
        self.log('acc', prec, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, seq_len, labels = batch
        data_1 = torch.FloatTensor(numpy.stack([audio],axis=0))

        # Spliited utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
        feats = numpy.stack(feats, axis = 0).astype(numpy.float)
        data_2 = torch.FloatTensor(feats).cuda()
        # Speaker embeddings
        with torch.no_grad():
            embedding_1, pho_out, seg_len = self.speaker_encoder.forward(data_1, aug = False)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2, pho_out, seg_len = self.speaker_encoder.forward(data_2, aug = False)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
        embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

    def configure_optimizers(self):
        return [self.optim], [self.scheduler]