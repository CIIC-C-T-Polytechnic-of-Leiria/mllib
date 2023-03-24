import numpy as np
from PIL import Image, ImageDraw
import base64
import json
import glob
import os
import io
import cv2

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