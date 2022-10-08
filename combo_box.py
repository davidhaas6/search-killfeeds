import vision
import cv2

def combine_boxes(boxes):
    midlines = []
    max_height = 0
    max_right = 0
    for box in boxes:
        average_y = (box[1] + box[3])/2
        midlines.append(average_y)
        if box[3] - box[1] > max_height:
            max_height = box[3] - box[1]
        if box[2] > max_right:
            max_right = box[2]
    clusters = cluster(midlines)
    left_alignment = []
    clust_midlines = []
    for clust in clusters:
        midline_thresh = 10
        padding = 10
        clust_avg = sum(clust)/len(clust)
        clust_midlines.append(clust_avg)
        furthest_left = 1000
        for box in boxes:
            average_y = (box[1] + box[3])/2
            if abs(average_y-clust_avg) <= midline_thresh:
                if box[0] <= furthest_left:
                    furthest_left = box[0]
        left_alignment.append(furthest_left-padding)
    # print(left_alignment)
    combo_boxes = []
    # startx, starty, endx, endy
    for i in range(len(left_alignment)):
        combo_boxes.append([int(left_alignment[i]),int(clust_midlines[i] - max_height/2),max_right,int(clust_midlines[i] + max_height/2)])
    return combo_boxes

def cluster(points):
    clusters = []
    eps = 10
    points_sorted = sorted(points)
    curr_point = points_sorted[0]
    curr_cluster = [curr_point]
    for point in points_sorted[1:]:
        if point <= curr_point + eps:
            curr_cluster.append(point)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [point]
        curr_point = point
    clusters.append(curr_cluster)
    return clusters

if __name__ == "__main__":
    det = vision.NNTextDetect('weights/east_text_detection_weights.pb')
    img = cv2.imread('example_killfeed.png')
    init_boxes,init_confidence = det.detect_text(img)
    det.draw_bboxes(img,init_boxes,True)
    combo_boxes = combine_boxes(init_boxes)
    det.draw_bboxes(img,combo_boxes,True)
