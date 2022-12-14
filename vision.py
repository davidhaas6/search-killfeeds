from typing import List
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pytesseract
import os
import tqdm
from imutils.object_detection import non_max_suppression
from PIL import Image


def thresholdize(rgb_img: np.ndarray, invert=False):
    """Preprocess an image into a binary image

    Args:
        img (ndarray): the image
        invert (bool, optional): Invert the binary image. Defaults to False.

    Returns:
        ndarray: The preprocessed image
    """
    img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,19,-2)
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if invert:
        img = cv2.bitwise_not(img)
    return img


def batch_preprocess(paths, out_dir=None, invert=False):
    """Preprocess a set of images

    Args:
        paths (List[str]): A list of image paths to read in
        out_dir (str, optional): A directory to write the preprocessed images to. Defaults to None.
        invert (bool, optional): Invert the binary images. Defaults to False.

    Returns:
        list[ndarray]: A list of the preprocessed images
    """
    imgs = [thresholdize(cv2.imread(p), invert) for p in paths]

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for i, im in tqdm.tqdm(enumerate(imgs)):
            file_path = os.path.join(out_dir, f"{i}.png")
            cv2.imwrite(file_path, im)

    return imgs


def topright_crop(img: np.ndarray, h_pct: float, w_pct: float) -> np.ndarray:
    """Crop the top right corner of an image

    Args:
        img (np.ndarray): The image to crop
        h_pct (float): The percent of img's height to crop the image to
        w_pct (float): The percent of img's width to crop the image to

    Returns:
        np.ndarray: the cropped image
    """
    h, w = img.shape[:2]
    return img[0 : int(h * h_pct), int(w * (1-w_pct)) : w]


def get_contour_bboxs(gray: np.ndarray, verbose=False):
    """Get the bounding boxes of an image via contour polygon approximation

    Args:
        gray (np.ndarray): A gray image
        verbose (bool, optional): Saves a debug image to ./out_contour.png. Defaults to False.

    Returns:
        List[tuple]: An array of (x,y,w,h) tuples representing bounding boxes
    """
    contours = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
    bboxes = []
    if verbose:
        output = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2RGB)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        if peri > 500:
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                bboxes.append(cv2.boundingRect(approx))
                if verbose:
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.rectangle(output, (x, y), (x + w, y + h), (36, 255, 12), 2)

    if verbose:
        cv2.imwrite(f"out_contour.png", output)

    return bboxes


def detect_text(img):
    return cv2.dnn.TextDetectionModel.detect(img)


def get_img_text(img, min_confidence=0) -> str:
    df = pytesseract.image_to_data(img, output_type="data.frame")
    text = " ".join(df.text[df.conf > min_confidence])
    return text



class TrOCR:
    """ Transformer-based OCR Model """
    def __init__(self,model_name='microsoft/trocr-small-printed') -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)


    def get_text(self, image: np.ndarray) -> str:
        image = Image.fromarray(image).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values 
        generated_ids = self.model.generate(pixel_values)
        textarr = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        return ' '.join(textarr)


class NNTextDetect:
    # https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV

    def __init__(self, model_path):
        self._net = cv2.dnn.readNet(model_path)

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        self._layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    def detect_text(self, rgb_img, min_confidence=0.5):
        """Returns bounding boxes of text in the image

        Args:
            rgb_img (ndarray): An image as a 3d array rgb channels
            min_confidence (float, optional): Drop results with confidence lower than this. Defaults to 0.5.

        Returns:
            (ndarray): Rows of bounding boxes represented as (startX, startY, endX, endY)
        """
        resized_img = self._resize_closest_multiple(rgb_img, 32)
        scores, geom = self._predict(resized_img)
        bboxes = self._geom_to_bboxes(geom, scores, min_confidence)

        if len(bboxes) > 0:
            # rescale bounding boxes to the original image's coordinate system
            h0, w0 = rgb_img.shape[:2]
            h1, w1 = resized_img.shape[:2]
            bboxes = bboxes * [h0 / h1, w0 / w1, h0 / h1, w0 / w1]
            bboxes = bboxes.round().astype('uint16')
        
        return bboxes

    def _predict(self, rgb_img) -> tuple:
        h, w = rgb_img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            rgb_img,
            1.0,
            (w, h),
            (123.68, 116.78, 103.94),
            swapRB=True,
        )
        self._net.setInput(blob)
        (scores, geometry) = self._net.forward(self._layer_names)
        return scores, geometry

    def _resize_closest_multiple(self, img: np.ndarray, multiple=32) -> np.ndarray:
        h, w = img.shape[:2]
        h1 = (h + multiple) - ((h + multiple) % multiple)
        w1 = (w + multiple) - ((w + multiple) % multiple)
        return cv2.resize(img, (w1, h1))

    def _geom_to_bboxes(self, geometry: np.ndarray, scores: np.ndarray, min_confidence=0.5) -> np.ndarray:
        (numRows, numCols) = geometry.shape[2:4]
        rects = []
        confidences = []

        for r in range(0, numRows):
            scoresData = scores[0, 0, r]
            boxData = geometry[0, :, r]
            # print(boxData)
            xData0 = geometry[0, 0, r]
            xData1 = geometry[0, 1, r]
            xData2 = geometry[0, 2, r]
            xData3 = geometry[0, 3, r]
            anglesData = geometry[0, 4, r]

            for c in range(0, numCols):
                if scoresData[c] < min_confidence:
                    continue

                (offsetX, offsetY) = (c * 4.0, r * 4.0)

                angle = anglesData[c]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[c] + xData2[c]
                w = xData1[c] + xData3[c]

                endX = int(offsetX + (cos * xData1[c]) + (sin * xData2[c]))
                endY = int(offsetY - (sin * xData1[c]) + (cos * xData2[c]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[c])

        boxes = non_max_suppression(np.array(rects), probs=confidences)

        return boxes

    def draw_bboxes(self, img, bboxes, verbose=False, color=(0,255,0)):
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        for (startX, startY, endX, endY) in bboxes:
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        if verbose:
            cv2.imshow("Text Detection", img)
            cv2.waitKey(0)
        return img
