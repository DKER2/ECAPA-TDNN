import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning import LightningModule, Trainer, seed_everything
from loss import AAMsoftmax, Phoneme_SSL_loss
from model import ECAPA_TDNN

class Task(LightningModule):
    def __init__( self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
        super().__init__()
        ## ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C = C).cuda()
        ## Classifier
        self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()
        self.phoneme_loss    = Phoneme_SSL_loss(num_frames=20, num_sample=3).cuda()

        self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
        self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)

        self.eval_list=kwargs['eval_list']
        self.eval_path=kwargs['eval_path']
    
    def training_step(self, batch, batch_idx):
        data, seq_len, labels = batch
        #labels = torch.LongTensor(labels)
        speaker_embedding, phonemes, seq_len = self.speaker_encoder.forward(data.cuda(), seq_len, aug = True)
        nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)
        loss_phn = self.phoneme_loss.forward(phonemes, seq_len)
        loss = loss_phn
        self.log('train_loss', loss, prog_bar=True)
        self.log('acc', prec, prog_bar=True)
        return loss

    def on_training_epoch_end(self):
        eval_network(self.eval_list, self.eval_path)

    def eval_network(self, eval_list, eval_path):
        self.speaker_encoder.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio, _  = soundfile.read(os.path.join(eval_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

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
                embedding_1, pho_out, seg_len = self.speaker_encoder.forward(data_1, [data_1.shape[1]], aug = False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2, pho_out, seg_len = self.speaker_encoder.forward(data_2, [data_2.shape[1] for i in range(data_2.shape[0])], aug = False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels  = [], []

        for line in lines:			
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))
            
        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        print("\nValidation error : ", EER)
        print("Validation minDCF : ", minDCF)
        return EER, minDCF

    def configure_optimizers(self):
        return [self.optim], [self.scheduler]