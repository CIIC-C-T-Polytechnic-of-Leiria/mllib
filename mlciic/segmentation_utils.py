#-------------------------------------------------------------------------------
# Utilitarian functions for segmentation tasks, file conversion, 
# video processing, etc.
#-------------------------------------------------------------------------------

#------------
# To Do List:
#------------
# 1. Translate to English
# 2. Add Docstrings
# 3. Test functions

import numpy as np
from PIL import Image, ImageDraw
import base64
import json
import glob
import os
import io
import cv2
from tdqm import tqdm
from rasterio.features import shapes  # gets shapes and values of regions 
from rasterio import Affine           # affine planar transformations
from shapely import wkt               # working with WKT files
from shapely.geometry import shape, MultiPolygon
from shapely.affinity import scale

def labelme_shapes_to_label(img_shape, shapes):
    """
    fonte: https://github.com/Jeff-sjtu/labelKeypoint/blob/master/labelme/utils.py
    """
    label_name_to_val = {'background': 0}
    lbl = np.zeros(img_shape[:2], dtype=np.int32)
    for shape in shapes:
        polygons = shape['points']
        label_name = shape['label']
        if label_name in label_name_to_val:
            label_value = label_name_to_val[label_name]
        else:
            label_value = len(label_name_to_val)
            label_name_to_val[label_name] = label_value
        mask = polygons_to_mask(img_shape[:2], polygons)
        lbl[mask] = label_value

    lbl_names = [None] * (max(label_name_to_val.values()) + 1)
    for label_name, label_value in label_name_to_val.items():
        lbl_names[label_value] = label_name
    return lbl, lbl_names

def polygons_to_mask(img_shape, polygons):
    """
    fonte: https://github.com/Jeff-sjtu/labelKeypoint/blob/master/labelme/utils.py
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def polygons_to_wkt(mask_paths, x_scale = 1, y_scale = 1):
    """
    Converts list of masks imgs to WKT string and rescales geometry, if 
    necessary 
    """
    wkt_list = list()
    for img_path in mask_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        polygon = mask_to_polygons(img)
        plgn_scld = scale(polygon, xfact=x_scale, yfact=y_scale, origin = (0,0))
        wkt_polygon = wkt.dumps(plgn_scld, rounding_precision = 0)
        wkt_list.append(wkt_polygon)
    wkt_to_file = '\n'.join(wkt_list)
    
    return wkt_to_file


def mask_to_polygons(mask_img):
    """
    Converts segmentation mask to shapely multipolygon.

    Adapted from: https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
    """
    all_polygons = list()
    
    for shp, value in shapes(source=mask_img.astype(np.uint8),mask=(mask_img>0), 
                             transform=Affine(1.0, 0, 0, 0, 1.0, 0)):
        all_polygons.append(shape(shp))

    all_polygons = MultiPolygon(all_polygons)

    # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
    # need to keep it a Multipolygon throughout
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    
    return all_polygons

def msks_paths_to_polygon_list(msks_paths):
    """
    Converts segmentation masks paths list to list of shapely multipolygons.
    """
    pol_list = list()
    for img_path in msks_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        polygon = mask_to_polygons(img)
        pol_list.append(polygon)
    return pol_list

def img_b64_to_array(img_b64):
    """
    fonte: https://github.com/Jeff-sjtu/labelKeypoint/blob/master/labelme/utils.py
    """
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(Image.open(f))
    return img_arr

def json2data(json_dir, out_dir_imgs, out_dir_mscs, img_size = (512, 512)):
    """
    Converte ficheiros json criado no Labelme em imagens e mascaras *.png

    Parâmetros:
        json_dir{str}: caminho para diretório com ficheiros *.json
        out_dir_imgs{str}: caminho de destino das imagens
        out_dir_mscs{str}: caminho de destino das máscaras
    """
    if not os.path.exists(out_dir_imgs):
        os.makedirs(out_dir_imgs)

    for filename in os.listdir(out_dir_imgs):
        if filename.endswith(".png"):
            file_path = os.path.join(out_dir_imgs, filename)
            os.remove(file_path)

    # cria lista de todos os ficheiros 'json'
    ficheiros = glob.glob(os.path.join(json_dir, "*.json"))
    print("A converter ficheiros json...")
    for ficheiro in ficheiros:
        # extrai dados do ficheiro json
        dados = json.load(open(ficheiro))
        # converte dados json em imagem
        img = img_b64_to_array(dados['imageData'])
        # converte poligonos de segmentacão em mascara
        msc_l, _ = labelme_shapes_to_label(img.shape, dados['shapes'])
        # redimensiona imagem e máscara
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_LINEAR)
        msc = cv2.resize(np.array(msc_l, dtype = 'uint8'), dsize=img_size)*255
        # extrai nome do ficheiro: 'frame_######'
        base = os.path.splitext(os.path.basename(ficheiro))[0]
        # extrai numero de indice de frame
        num_frame = base.split('_')[1]
        # salva img (em modo RGB!!) e masc en ficheiro *.png 
        cv2.imwrite(os.path.join(out_dir_imgs, base + ".png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir_mscs, 'masc_' + str(num_frame) + ".png"), msc)
    print(f"Fim de conversão. Foram criados {len(ficheiros)} imagens e respetivas mascaras.")

def frame2video(img_list, nome_ficheiro='video'):
    """ 
    Converte lista de imagens em ficheiro AVI com a mesma resolucão da primeira 
    imagem da lista.

      Parametros: - lista de imagens PNG, TIFF, JPEG, BMP, WEBP, STK, LSM ou XCF
                  - nome do ficheiro do video

      Devolve: salva vídeo no diretório de execucão
    """
    # guarda dimensões da primeira imagem
    img = cv2.imread(img_list[0])
    height, width, layers = img.shape
    size = (width, height)

    img_array = list()
    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        img_array.append(img)

    video = cv2.VideoWriter(filename= nome_ficheiro + '.avi',
                            fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=25,
                            frameSize=size)

    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()

def vid2frame(vid_caminho, frames_dir, altura_imgs = 128, largura_imgs = 128, verbose = False):
    """
    Converte video em lista de frames no formato PNG
    """
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    for filename in os.listdir(frames_dir):
        if filename.endswith(".png"):
            file_path = os.path.join(frames_dir, filename)
            os.remove(file_path)

    # cria objeto de captura de video
    vid_cap = cv2.VideoCapture(vid_caminho)

    # guarda dimensões video original
    largura_orig = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_orig = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if verbose == False:
        # escreve propriedades do do video e dos frames a criar
        print(
        f"""
        {'-'*80}
        # \033[1mPropriedades do vídeo original\033[0m
        # Largura: {largura_orig}, Altura: {altura_orig}
        # Frames por segundo: {round(vid_cap.get(cv2.CAP_PROP_FPS), 2)}
        {'-'*80}
        # \033[1mPropriedades dos frames resultantes\033[0m
        # Largura: {largura_imgs}, Altura: {altura_imgs}
        # Número total de frames: {int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        # Frames guardados no diretório: {frames_dir}
        {'-'*80}
        """
        )

    if vid_cap.isOpened():
        frame_num = 0
        while (True):
            # lê vídeo frame a frame
            success, frame = vid_cap.read()
            # pára ciclo
            if success == False:
            #print("Fim de captura do vídeo")
                break
            # redimensiona frames
            frame_red = cv2.resize(frame, (largura_imgs, altura_imgs))
            # guarda em formato RGB (em vez de BGR!)
            cv2.imwrite(frames_dir + "/frame_" + str(frame_num).zfill(6) + ".png", frame_red)
            frame_num += 1
    print('Conversão concluída')
    return (altura_orig, largura_orig)

def wkt2masc(wkt_file, images_path, orig_dims, height, width):
    """ 
    Converts WKT files to segmentation masks.
    Parameters:
        wkt_file {str} -- path to the WKT file
        images_path {str} -- path to the folder where the masks will be saved
        orig_dims {tuple} -- original dimensions of the masks
        height {int} -- desired height of the masks
        width {int} -- desired width of the masks
    Returns:
        Creates PNG images of the masks
    """

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # delete files in the folder, if any
    for filename in os.listdir(images_path):
        if filename.endswith(".png"):
            file_path = os.path.join(images_path, filename)
            os.remove(file_path)

    # open WKT file
    wkt = open(wkt_file, 'r')
    num_lines = len(wkt.readlines())
    cnt = 0
    print(f"""
    {'-'*38}
    # \033[1mProperties of the resulting masks\033[0m
    # Width: {width}, Height: {height}
    # Number of masks to create: {num_lines}
    {'-'*38}
    """)

    pbar = tqdm(total=num_lines)

    # process each line of the WKT file
    with open(wkt_file) as wkt:
        for line in wkt:
            # extract numbers from the line
            points = [int(s) for s in re.findall('[0-9]+', line)]
            # create empty mask
            # mask = np.zeros((height, width), dtype=np.uint8)
            mask = np.zeros(orig_dims)
            # create array with polygon points, with 2 columns (x,y)
            arr = np.zeros((int(len(points)/2), 2))

            # fill array 'arr'
            j = 0
            for i in range(0, int(len(points)/2)):
                arr[i, 0] = int(points[j])
                arr[i, 1] = int(points[j+1])
                j += 2
            # draw mask
            cv2.drawContours(image=mask,
                             contours=[arr.astype(np.int32)],
                             contourIdx=-1,
                             color=(255, 255, 255),
                             thickness=-1,  # if > 0, thickness of the contour; if -1, fill object
                             lineType=cv2.LINE_AA)
            # resize frames
            mask_resized = cv2.resize(mask, (width, height))
            cv2.imwrite(images_path + "/mask_" +
                        str(cnt).zfill(6) + ".png", mask_resized)
            cnt += 1
            pbar.update(1)

    pbar.close()
    wkt.close()