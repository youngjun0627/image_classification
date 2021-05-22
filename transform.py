import albumentations
from albumentations.pytorch import ToTensor

def create_train_transform(flip,\
        noise,\
        cutout,\
        resize,\
        size = 224):
    translist=[]
    ### resize
    if resize:
        translist+=[albumentations.Resize(size+30,size+30)] ## original image width : 300
        translist+=[albumentations.RandomCrop(size,size,always_apply=True)]
    if flip:
        translist+=[albumentations.OneOf([
            albumentations.HorizontalFlip(),
            albumentations.RandomRotate90(),
            albumentations.VerticalFlip()],p=0.5)]

    ### noise
    if noise:
        translist+=[albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0,30.0))], p=0.5)]
    ### cutout
    if cutout:
        translist+= [albumentations.Cutout(max_h_size = int(size * 0.2), max_w_size = int(size * 0.2), num_holes = 1,p=0.3)]

    ### normalized & totensor
    translist+=[albumentations.Normalize(mean = (0.4875, 0.5115, 0.5364), std = (0.2432, 0.2359, 0.2503))] # 위에서 구한 값으로 
    translist+=[ToTensor()]
    transform = albumentations.Compose(translist)
    return transform

def create_validation_transform(resize,\
        size = 224):
    translist=[]
    ### resize
    if resize:
        translist+=[albumentations.Resize(size,size)]                                                                                    
    ### normalized & totensor
    translist+=[albumentations.Normalize(mean = (0.4859, 0.5137, 0.5416), std = (0.2178, 0.2138, 0.2253))]
    translist+=[ToTensor()]
    transform = albumentations.Compose(translist)
    return transform


