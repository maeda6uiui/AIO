import argparse
import logging
import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torchvision

#Object detection
import cv2
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import hashing

#Setup Detectron2
setup_logger()

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
if torch.cuda.is_available()==False:
    cfg.MODEL.DEVICE="cpu"

predictor = DefaultPredictor(cfg)

normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

#Set up a logger.
logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PredRegion(object):
    def __init__(self,region,c):
        self.region=region
        self.c=c

def get_pred_regions_as_images(im_dir):
    regions=[]

    files = os.listdir(im_dir)
    for file in files:
        im_filepath=os.path.join(im_dir,file)

        try:
            image_pil=Image.open(im_filepath)
        except:
            logger.error("Image file open error: {}".format(im_filepath))
            continue

        image_cv2 = cv2.imread(im_filepath)
        outputs = predictor(image_cv2)

        pred_boxes_tmp = outputs["instances"].pred_boxes.tensor
        for i in range(pred_boxes_tmp.size()[0]):
            top_left_x=int(pred_boxes_tmp[i][0])
            top_left_y=int(pred_boxes_tmp[i][1])
            bottom_right_x=int(pred_boxes_tmp[i][2])
            bottom_right_y=int(pred_boxes_tmp[i][3])

            image_region=image_pil.crop((top_left_x,top_left_y,bottom_right_x,bottom_right_y))
            width,height=image_pil.size
            c=(
                top_left_x/width,
                top_left_y/height,
                bottom_right_x/width,
                bottom_right_y/height,
                (bottom_right_x-top_left_x)*(bottom_right_y-top_left_y)/(width*height)
            )

            region=PredRegion(image_region,c)
            regions.append(region)

    return regions

def create_im_embedding(pred_region,vgg16,fc_vgg16,fc_c):
    region=pred_region.region
    c=pred_region.c

    region=region.convert("RGB")
    region_tensor = preprocess(region).unsqueeze(0).to(device)

    im_embedding=vgg16(region_tensor).to(device)
    im_embedding=fc_vgg16(im_embedding)

    c_tensor=torch.empty(1,5,dtype=torch.float).to(device)
    for i in range(5):
        c_tensor[0,i]=c[i]

    c_tensor=fc_c(c_tensor)

    return im_embedding+c_tensor   #Add im_embedding and c. ref: ImageBERT

def create_im_embeddings(pred_regions,vgg16,embedding_dim,fc_vgg16,fc_c):
    ret=torch.empty(0,embedding_dim,dtype=torch.float).to(device)

    for pred_region in pred_regions:
        tmp=create_im_embedding(pred_region,vgg16,fc_vgg16,fc_c)
        ret=torch.cat([ret,tmp],dim=0)

    return ret

def main(im_base_dir,embedding_dim,embeddings_save_dir):
    #Load the article list.
    article_list_filepath=os.path.join(im_base_dir,"article_list.txt")
    df = pd.read_table(article_list_filepath, header=None)

    articles={}
    for row in df.itertuples(name=None):
        article_name = row[1]
        dir_1 = row[2]
        dir_2 = row[3]

        article_hash=hashing.get_md5_hash(article_name)

        im_dir = os.path.join(im_base_dir,"Images",str(dir_1),str(dir_2))
        articles[article_hash]=im_dir

    #Create a directory to save the image embeddings in.
    os.makedirs(embeddings_save_dir,exist_ok=True)

    #Create a VGG16 model.
    vgg16=torchvision.models.vgg16(pretrained=True)
    vgg16.to(device)
    vgg16.eval()

    fc_vgg16=nn.Linear(1000,embedding_dim).to(device)
    fc_c=nn.Linear(5,embedding_dim).to(device)

    #Create image embeddings.
    for article_hash,im_dir in tqdm(articles.items()):
        pred_regions=get_pred_regions_as_images(im_dir)
        im_embeddings=create_im_embeddings(pred_regions,vgg16,embedding_dim,fc_vgg16,fc_c)

        embeddings_save_filepath=os.path.join(embeddings_save_dir,article_hash+".pt")
        torch.save(im_embeddings,embeddings_save_filepath)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="AIO")

    parser.add_argument("--im_base_dir",type=str,default="~/WikipediaImages")
    parser.add_argument("--embedding_dim",type=int,default=768)
    parser.add_argument("--embeddings_save_dir",type=str,default="~/VGG16Embeddings")

    args=parser.parse_args()

    logger.info("im_base_dir: {}".format(args.im_base_dir))
    logger.info("embedding_dim: {}".format(args.embedding_dim))
    logger.info("embeddings_save_dir: {}".format(args.embeddings_save_dir))

    main(args.im_base_dir,args.embedding_dim,args.embeddings_save_dir)
