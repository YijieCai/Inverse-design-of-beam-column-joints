from PIL import Image, ImageSequence
from tqdm import tqdm



for section_num in tqdm(range(1,5)):
    image_paths = []
    for i in range(100):
        image_paths.append(f'./02Section/{i}_{section_num}.png')
    images = [Image.open(image) for image in image_paths]
    images[0].save(f'./05Gif_Section/Section_{section_num}.gif', format='GIF', append_images=images,
                   save_all=True, duration=0.1, loop=0)


image_paths = []
for i in range(100):
    image_paths.append(f'./01Img/DDPM_Section_{i*10}.png')
images = [Image.open(image) for image in image_paths]
images[0].save(f'./05Gif_Section/Section_DDPM.gif', format='GIF', append_images=images,
               save_all=True, duration=0.1, loop=0)



