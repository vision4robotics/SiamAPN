from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch.nn.functional as F
from pysot.core.config_adapn import cfg
from pysot.tracker.base_tracker import SiameseTracker

class ADSiamAPNTracker(SiameseTracker):
    def __init__(self, model):
        super(ADSiamAPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 
            
        self.anchor_num=1
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.model = model
        self.model.eval()

    def generate_anchor(self):
  
        
        mapp=self.model.ranchors        
        size=cfg.TRAIN.OUTPUT_SIZE

               
        x=np.tile((cfg.ANCHOR.STRIDE*(np.linspace(0,size-1,size)))-cfg.ANCHOR.STRIDE*(size-1)/2,size).reshape(-1)
        y=np.tile((cfg.ANCHOR.STRIDE*(np.linspace(0,size-1,size))).reshape(-1,1)-cfg.ANCHOR.STRIDE*(size-1)/2,size).reshape(-1)
        shap=self.con(mapp[0]).cpu().detach().numpy()
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))     
        w=shap[0,yy,xx]+shap[1,yy,xx]
        h=shap[2,yy,xx]+shap[3,yy,xx]
        x=x-shap[0,yy,xx]+w/2
        y=y-shap[2,yy,xx]+h/2

        anchor=np.zeros((size**2,4))
  
        anchor[:,0]=x
        anchor[:,1]=y
        anchor[:,2]=w
        anchor[:,3]=h
        
       
        return anchor
    
    
    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.image=img
        
        
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        
        
        self.size = np.array([bbox[2], bbox[3]])
        self.firstbbox=np.concatenate((self.center_pos,self.size))
        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.template=z_crop

    
        self.model.template(z_crop)

  
    def con(self, x):
        return  x*(cfg.TRAIN.SEARCH_SIZE//4)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRAIN.EXEMPLAR_SIZE / s_z

        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        

        outputs = self.model.track(x_crop)
        
        self.anchors = self.generate_anchor() 
        score1 = self._convert_score(outputs['cls1'])*cfg.TRACK.w1
        score2 = self._convert_score(outputs['cls2'])*cfg.TRACK.w2
        score3=(outputs['cls3']).view(-1).cpu().detach().numpy()*cfg.TRACK.w3
        score=(score1+score2+score3)/3 


            
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

       
        def change(r):
            
            return np.maximum(r, 1. / (r+1e-5))

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/(self.size[1]+1e-5)) /
                     (pred_bbox[2, :]/(pred_bbox[3, :]+1e-5)))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        
        bbox = pred_bbox[:, best_idx] / scale_z
        
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR 

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
      
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        best_score = score[best_idx]

        return {
                'bbox': bbox,
                'best_score': best_score,
               }