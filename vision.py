from matplotlib import pyplot as plt
import cv2 
import numpy as np
import pytesseract
import os
import tqdm

def preprocess(img,invert=False):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if invert: img = cv2.bitwise_not(img)
    return img



def batch_preprocess(paths, out_dir=None, invert=False):
    imgs = [preprocess(cv2.imread(p)) for p in paths]

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for i, im in tqdm.tqdm(enumerate(imgs)):
            file_path = os.path.join(out_dir, f'{i}.png')
            cv2.imwrite(file_path,im)

    return imgs


def crop_topright_quadrant(img):
    h, w, = img.shape[:2]
    return img[0:h//2, w//2:w]


def get_contour_bboxs(gray: np.ndarray, verbose=False):
    contours = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
    bboxes = []
    if verbose: 
        output = cv2.cvtColor(gray.copy(),cv2.COLOR_GRAY2RGB)
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        if peri > 500:
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                bboxes.append(cv2.boundingRect(approx))
                if verbose:
                    x,y,w,h = cv2.boundingRect(approx)
                    cv2.rectangle(output,(x,y),(x+w,y+h),(36,255,12),2)

    if verbose: 
        cv2.imwrite(f'out_contour.png',output)

    return bboxes


def get_img_text(img, min_confidence=0):
    df = pytesseract.image_to_data(img, output_type="data.frame")
    text = " ".join(df.text[df.conf > min_confidence])
    return text
