import cv2

from AWing.landmark_api import get_AWing_landmark_model, preprocess_image_for_landmarks

landmark_model, _, landmark_model_prediction_fn, landmark_count = get_AWing_landmark_model()

image = cv2.imread('images/leo128x128.png')

preprocessed_image = preprocess_image_for_landmarks(image, transpose=True)

pred_landmarks, pred_landmark_confs = landmark_model_prediction_fn(preprocessed_image)

for l in pred_landmarks[0]:
    p = (l + 0.5).astype(int)
    cv2.circle(image, (p[0], p[1]), 3, (0, 0, 255), -1)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()