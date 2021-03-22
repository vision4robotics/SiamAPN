# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config_apn import cfg
from pysot.models.loss_apn import select_cross_entropy_loss, weight_l1_loss,l1loss,IOULoss
from pysot.models.backbone.alexnet import AlexNet
from pysot.models.utile_apn import APN,clsandloc
import numpy as np
from pysot.datasets.anchortarget_apn import AnchorTarget3


class ModelBuilderAPN(nn.Module):
    def __init__(self):
        super(ModelBuilderAPN, self).__init__()

        self.backbone = AlexNet().cuda()
        self.grader=APN(cfg).cuda()
        self.new=clsandloc(cfg).cuda()

        self.fin2=AnchorTarget3() 
        self.cls3loss=nn.BCEWithLogitsLoss()
        self.IOULOSS=IOULoss()
             
        
    def template(self, z):

        zf1,zf = self.backbone(z)

        self.zf=zf
        
        self.zf1=zf1

    
    def track(self, x):
        
        xf1,xf = self.backbone(x)  
        xff,ress=self.grader(xf1,self.zf1)    

        self.ranchors=xff
              
        cls1,cls2,cls3,loc =self.new(xf,self.zf,ress)  
 
        return {
                'cls1': cls1,
                'cls2': cls2,
                'cls3': cls3,
                'loc': loc
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)

        return cls


    def getcenter(self,mapp):

        def con(x):
            return x*143
        
        size=mapp.size()[3]
        #location 
        x=np.tile((cfg.TRAIN.STRIDE*(np.linspace(0,size-1,size))+cfg.TRAIN.MOV)-cfg.TRAIN.SEARCH_SIZE//2,size).reshape(-1)
        y=np.tile((cfg.TRAIN.STRIDE*(np.linspace(0,size-1,size))+cfg.TRAIN.MOV).reshape(-1,1)-cfg.TRAIN.SEARCH_SIZE//2,size).reshape(-1)
        shap=con(mapp).cpu().detach().numpy()
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))

        w=shap[:,0,yy,xx]+shap[:,1,yy,xx]
        h=shap[:,2,yy,xx]+shap[:,3,yy,xx]
        x=x-shap[:,0,yy,xx]+w/2
        y=y-shap[:,2,yy,xx]+h/2
        
        anchor=np.zeros((cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,size**2,4))

        anchor[:,:,0]=x+cfg.TRAIN.SEARCH_SIZE//2
        anchor[:,:,1]=y+cfg.TRAIN.SEARCH_SIZE//2
        anchor[:,:,2]=w
        anchor[:,:,3]=h


        return anchor


    
    def _convert_bbox(self, delta, anchor):
        delta = delta.contiguous().view(anchor.shape[0],4, -1)
        
        anchor=t.Tensor(anchor).cuda().float()
        locc=t.zeros_like(anchor).cuda()
        
        locc[:,:,0] = delta[:,0, :] * anchor[:,:, 2] + anchor[:, :,0]
        locc[:,:,1] = delta[:,1, :] * anchor[:,:, 3] + anchor[:,:, 1]
        locc[:,:,2] = t.exp(delta[:,2, :]) * anchor[:,:, 2]
        locc[:,:,3] = t.exp(delta[:,3, :]) * anchor[:, :,3]
        
        loc=t.zeros_like(anchor).cuda()
        loc[:,:,0]=locc[:,:,0]-locc[:,:,2]/2
        loc[:,:,1]=locc[:,:,1]-locc[:,:,3]/2
        loc[:,:,2]=locc[:,:,0]+locc[:,:,2]/2
        loc[:,:,3]=locc[:,:,1]+locc[:,:,3]/2
        
        return loc

    
    def forward(self,data):
        """ only used in training
        """
                
        template = data['template'].cuda()
        search =data['search'].cuda()
        bbox=data['bbox'].cuda()
        labelcls2=data['label_cls2'].cuda()
        labelxff=data['labelxff'].cuda()
        weightcls3=data['weightcls3'].cuda()
        labelcls3=data['labelcls3'].cuda()
        weightxff=data['weightxff'].cuda()
        

        
        zf1,zf = self.backbone(template)
        xf1,xf = self.backbone(search)
        xff,ress=self.grader(xf1,zf1)
        
        anchors=self.getcenter(xff) 

        label_cls,label_loc,label_loc_weight\
            =self.fin2.get(anchors,bbox,xff.size()[3])


        

        
        cls1,cls2,cls3,loc=self.new(xf,zf,ress)
        
        cls1 = self.log_softmax(cls1)  
        cls2 = self.log_softmax(cls2) 

        
        cls_loss1 = select_cross_entropy_loss(cls1, label_cls)
        cls_loss2 = select_cross_entropy_loss(cls2, labelcls2)
        cls_loss3 = self.cls3loss(cls3, labelcls3)  

        cls_loss= cfg.TRAIN.w3*cls_loss3 + cfg.TRAIN.w1*cls_loss1 + cfg.TRAIN.w2*cls_loss2

        loc_loss1 = weight_l1_loss(loc, label_loc, label_loc_weight) 


        pre_bbox=self._convert_bbox(loc,anchors)
        label_bbox=self._convert_bbox(label_loc,anchors)
        
        loc_loss2=self.IOULOSS(pre_bbox,label_bbox,label_loc_weight)

        loc_loss=cfg.TRAIN.w4*loc_loss1+cfg.TRAIN.w5*loc_loss2
        
        shapeloss=l1loss(xff,labelxff,weightxff) 
        
        

        outputs = {}
        outputs['total_loss'] =\
            cfg.TRAIN.LOC_WEIGHT*loc_loss\
                +cfg.TRAIN.CLS_WEIGHT*cls_loss\
                    +cfg.TRAIN.SHAPE_WEIGHT*shapeloss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['shapeloss'] = shapeloss
                                                   #2 4 1  都用loss2

        return outputs
    
    

    



    

    
     


  
    

