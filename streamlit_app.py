#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import os
import seaborn as sns
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
import datasets
import torchvision
from datasets import Features
from torch.utils.data import DataLoader,Dataset

#######################
# Page configuration
st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("## Dataset")


ressource_path = r"."

crop_path = os.path.join(ressource_path, "Crop_img")
img_path = os.path.join(ressource_path, "Images")
ann_path = os.path.join(ressource_path, "Annotation")

dt_src = pd.read_csv(os.path.join(ressource_path, 'src_file.csv'), index_col=0).sample(frac=1, random_state=0, ignore_index=True)
dt_src.race = dt_src.race.str.split('-').str[1]

st.dataframe(dt_src)

st.markdown("## Describe")
st.dataframe(dt_src.describe())

st.markdown("## Race")
st.dataframe(dt_src['race'].unique())




my_listold = ['Irish_wolfhound', 'Pomeranian', 'Afghan_hound',
       'Scottish_deerhound', 'Bernese_mountain_dog', 'Maltese_dog',
       'Airedale', 'Saluki', 'Shih', 'cairn']

my_list = ['Irish_wolfhound', 'cairn', 'Afghan_hound', 'Pomeranian']

option = st.selectbox('select dog race',     ('Irish_wolfhound', 'cairn', 'Afghan_hound', 'Pomeranian'))

st.write('You selected:', option)


path_omg = rd.choice(my_list)
print(path_omg)

list_omg = dt_src[dt_src.race==path_omg]


list_omg = list_omg['image_path']
list_omg.reset_index()
img_rd = rd.choice(list_omg.tolist())
#img_rd = rd.choice(list_omg)

st.image(img_rd)


st.markdown("## Count plot")

countrace= sns.countplot(dt_src['race'])

st.pyplot(countrace.get_figure())



st.markdown("## Crooping")
dir_img_list = ['n02090721-Irish_wolfhound', 'n02096177-cairn', 'n02088094-Afghan_hound']
def extract_bounding_box_values(breed, dog):
    xml_file = os.path.join(ann_path, breed, dog)

    # Parsing du fichier XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Récupération des valeurs xmin, xmax, ymin, ymax
    xmin = int(root.find('object').find('bndbox').find('xmin').text)
    ymin = int(root.find('object').find('bndbox').find('ymin').text)
    xmax = int(root.find('object').find('bndbox').find('xmax').text)
    ymax = int(root.find('object').find('bndbox').find('ymax').text)

    return xmin, ymin, xmax, ymax

def crop_image(breed, dog):
    img_file = os.path.join(img_path, breed, dog + '.jpg')
    img = Image.open(img_file)
    img_array = np.array(img)

    xmin, ymin, xmax, ymax = extract_bounding_box_values(breed, dog)
    img_crop = img_array[ymin:ymax, xmin:xmax, :]

    return img_crop

    
plt.figure(figsize=(20, 20))
for i in range(3):
    rd_breed = dir_img_list[i]
    random_breed_path = os.path.join(img_path, rd_breed)
    img_list = os.listdir(random_breed_path)

    rd_dog = rd.choice(img_list)
    random_img_path = os.path.join(random_breed_path, rd_dog)

    xmin, ymin, xmax, ymax = extract_bounding_box_values(rd_breed, rd_dog[:-4])

    img = np.array(Image.open(random_img_path))
    crop_img = crop_image(rd_breed, rd_dog[:-4])

    ## Create subplots explicitly
    plt.subplot(4, 2, 2 * i + 1)
    plt.imshow(img)
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color='cornflowerblue')
    plt.title("{} {}".format(rd_breed, img.shape))

    plt.subplot(4, 2, 2 * i + 2)
    plt.imshow(crop_img)
    plt.title("{}".format(crop_img.shape))

plt.show()

fig = plt.show()
st.pyplot(fig)


st.markdown("## Data augmentation for train")



features = datasets.Features({
    "img": datasets.Image(),
    "label": datasets.ClassLabel(names=my_list),
})
data = {
        "img": list(dt_src["image_path"]),
        "label": list(dt_src["race"])
        }

print(set(data["label"]))

ds = datasets.Dataset.from_dict(data, features=features)
ds = ds.train_test_split(test_size=0.2, stratify_by_column='label', shuffle=True, seed=0)
train = ds['train']
test = ds['test']
test
test = test.train_test_split(test_size=0.5, stratify_by_column='label', shuffle=True, seed=0)
val = test['train']

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    ColorJitter,
    RandomPerspective,
    RandomRotation,
)

normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_transforms = Compose(
        [
            RandomResizedCrop(224),
            ColorJitter(brightness=0.5, contrast=0.1, saturation=0.3,hue=0.1),
            RandomPerspective(distortion_scale=0.6, p=1.0),
            RandomRotation(degrees=(0, 45)),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

img_list = ['Crop_img/n02088094_26.jpg', 'Crop_img/n02088094_60.jpg', 'Crop_img/n02088094_93.jpg','Crop_img/n02088094_115.jpg']

class MyImageDataset(Dataset):
    
  def __init__(self,image_list,transforms=None):
    self.image_list=image_list
    self.transforms=transforms
    
  def __len__(self):
    return len(self.image_list)

  def __getitem__(self,i):
    img=plt.imread(self.image_list[i])
    img=Image.fromarray(img).convert('RGB')
    img=np.array(img).astype(np.uint8)

    if self.transforms is not None:
      img=self.transforms(img)
    return img

def show_batch(dataloader, rows, columns):
    data=iter(dataloader)
    fig = plt.figure(figsize=(15, 12))
    
    imgs=next(data)

    for i in range(rows*columns):
        npimg=imgs[i].numpy()
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.axis('off')
    plt.show()  


transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToPILImage(), 
                              torchvision.transforms.Resize((224,224)),
                              torchvision.transforms.ToTensor(),
                              ])
sample_ds=MyImageDataset(img_list,transform)
sample_dl=DataLoader(sample_ds,batch_size=4) # , shuffle=True
fig = show_batch(sample_dl, 2, 2)

st.markdown("## Resize")
st.pyplot(fig)

transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToPILImage(), 
                              torchvision.transforms.RandomResizedCrop(224),
                              torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.3,hue=0.1),
                              torchvision.transforms.ToTensor(),
                              ])
sample_ds=MyImageDataset(img_list,transform)
sample_dl=DataLoader(sample_ds,batch_size=4) # , shuffle=True
fig = show_batch(sample_dl, 2, 2)

st.markdown("## ColorJitter")
st.pyplot(fig)

transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToPILImage(), 
                              torchvision.transforms.Resize((224,224)),
                              torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                              torchvision.transforms.ToTensor(),
                              ])
sample_ds=MyImageDataset(img_list,transform)
sample_dl=DataLoader(sample_ds,batch_size=4) # , shuffle=True
fig = show_batch(sample_dl, 2, 2)

st.markdown("## RandomPerspective")
st.pyplot(fig)

transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToPILImage(), 
                              torchvision.transforms.RandomResizedCrop(224),
                              torchvision.transforms.RandomRotation(degrees=(0, 45)),
                              torchvision.transforms.ToTensor(),
                              ])
sample_ds=MyImageDataset(img_list,transform)
sample_dl=DataLoader(sample_ds,batch_size=4) # , shuffle=True
fig = show_batch(sample_dl, 2, 2)

st.markdown("## RandomRotation")
st.pyplot(fig)

transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToPILImage(), 
                              torchvision.transforms.RandomResizedCrop(224),
                              torchvision.transforms.RandomHorizontalFlip(),
                              torchvision.transforms.ToTensor(),
                              ])
sample_ds=MyImageDataset(img_list,transform)
sample_dl=DataLoader(sample_ds,batch_size=4) # , shuffle=True
fig = show_batch(sample_dl, 2, 2)

st.markdown("## RandomHorizontalFlip")
st.pyplot(fig)

transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToPILImage(), 
                              torchvision.transforms.RandomResizedCrop(224),
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                              ])
sample_ds=MyImageDataset(img_list,transform)
sample_dl=DataLoader(sample_ds,batch_size=4) # , shuffle=True
fig = show_batch(sample_dl, 2, 2)

st.markdown("## Normalize")
st.pyplot(fig)

st.markdown("## Images prédiction")


st.markdown("## Irish_wolfhound")
st.image(os.path.join(ressource_path, 'imgpred/imgpred1.jpg'))

st.markdown("## cairn")
st.image(os.path.join(ressource_path, 'imgpred/imgpred2.jpg'))

st.markdown("## Scottish_deerhound")
st.image(os.path.join(ressource_path, 'imgpred/imgpred3.jpg'))


#sns.countplot(dt_src['race'])
