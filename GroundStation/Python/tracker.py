import numpy as np

class SimpleTracker:
    """Simple object tracker using centroid tracking"""
    def __init__(self, max_disappeared=5, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # object_id: {'centroid': (x, y), 'bbox': [x1, y1, x2, y2], 'class': label, 'confidence': score}
        self.disappeared = {}  # object_id: frame_count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, class_id, confidence):
        """Register a new object"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'class': class_id,
            'confidence': confidence
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections=None):
        """Update tracker with new detections or predict positions"""
        if detections is None or len(detections) == 0:
            # No detections - just update disappeared counter
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Convert detections to centroids
        input_centroids = []
        input_bboxes = []
        input_classes = []
        input_confidences = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))
            input_bboxes.append([x1, y1, x2, y2])
            input_classes.append(detection['class'])
            input_confidences.append(detection['confidence'])

        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], input_classes[i], input_confidences[i])
        else:
            # Compute distance matrix between existing objects and new detections
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())

            # Compute distances
            distances = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids), axis=2)

            # Find minimum distances
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if distances[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    # Update object
                    self.objects[object_id]['centroid'] = input_centroids[col]
                    self.objects[object_id]['bbox'] = input_bboxes[col]
                    self.objects[object_id]['class'] = input_classes[col]
                    self.objects[object_id]['confidence'] = input_confidences[col]
                    self.disappeared[object_id] = 0

                    used_row_indices.add(row)
                    used_col_indices.add(col)

            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, distances.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, distances.shape[1])).difference(used_col_indices)

            # If more objects than detections, mark objects as disappeared
            if distances.shape[0] >= distances.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            # Register new objects
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col], input_bboxes[col], input_classes[col], input_confidences[col])

        return self.objects

    def get_predictions(self):
        """Get current tracked objects as detection format"""
        predictions = []
        for obj_id, obj_data in self.objects.items():
            x1, y1, x2, y2 = obj_data['bbox']
            predictions.append({
                'bbox': [x1, y1, x2, y2],
                'class': obj_data['class'],
                'confidence': obj_data['confidence'],
                'track_id': obj_id
            })
        return predictions