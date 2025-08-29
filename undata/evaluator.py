from typing import Any, List

from typing_extensions import Tuple

from undata import UNDataset
from undata.unbbox import UNBBox
from undata.unsample import UNSample


class Evaluator:
    def __init__(self):
        pass

    @staticmethod
    def _iou_xyxy(box1: UNBBox, box2: UNBBox) -> float:
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1: UNBBox in xyxy format.
            box2: UNBBox in xyxy format.

        Returns:
            The IoU value (float) between 0 and 1.
        """
        # Check format, it must be xyxy
        if box1.format != "xyxy" or box2.format != "xyxy":
            raise ValueError("Bounding Box must be in xyxy format.")

        # Convert UNBBox to xyxy coordinates for calculation
        x1_1, y1_1, x2_1, y2_1 = (
            box1.coords[0],
            box1.coords[1],
            box1.coords[2],
            box1.coords[3],
        )
        x1_2, y1_2, x2_2, y2_2 = (
            box2.coords[0],
            box2.coords[1],
            box2.coords[2],
            box2.coords[3],
        )

        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0  # No intersection

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    # def _bbox_iou(self, gt_bbox: UNBBox, predicted_bbox: UNBBox, img_width: int, img_height: int) -> float:
    #    # Convert both bboxes to xyxy format
    #    gt_sample = UNSample(bbox=[gt_bbox], image_w=img_width, image_h=img_height)
    #    gt_sample = gt_sample.bbox_convert(to_format="xyxy", inplace=False)
    #
    #    predicted_sample = UNSample(bbox=[predicted_bbox], image_w=img_width, image_h=img_height)
    #    predicted_sample = predicted_sample.bbox_convert(to_format="xyxy", inplace=False)
    #
    #    return self._iou_xyxy(gt_sample.bbox[0], predicted_sample.bbox[0])

    @staticmethod
    def sample_iou(
        gt_sample: UNSample,
        predicted_sample: UNSample,
        iou_threshold_for_match: float = 0.5,
        score_threshold: float = 0.5,
        check_label: bool = False,
    ) -> Tuple[List[Any], List[Any]] | Tuple[List[List[bool]], List[List[float]]]:
        """
        Compute Intersection over Union (IoUs) between ground truth and predicted bounding boxes
        and filters them based on a threshold.

        Args:
            gt_sample: Ground truth UNSample object.
            predicted_sample: Predicted UNSample object.
            iou_threshold_for_match: IoU threshold for filtering.
            score_threshold: threshold on bounding box score for filtering.
            check_label: Check if label of predicted bounding box and ground truth bounding box match.

        Returns:
            A tuple containing:
            - A list of IoU values.
            - A list of booleans indicating whether each IoU is above the threshold.
        """
        if not gt_sample.bbox:
            return [], []  # No ground truth boxes

        gt_sample.bbox_convert(to_format="xyxy")
        predicted_sample.bbox_convert(to_format="xyxy")

        # Create a matrix with ground truth on rows and predictions on columns and a matrix with bools based dn threshold
        ious_matrix = []
        ious_matrix_bool = []

        for gt_bbox in gt_sample.bbox:
            ious_single_gt = []
            ious_single_gt_bool = []

            for predicted_bbox in predicted_sample.bbox:
                iou = Evaluator._iou_xyxy(gt_bbox, predicted_bbox)
                ious_single_gt.append(iou)

                filter_result = (
                    iou > iou_threshold_for_match
                    and predicted_bbox.score > score_threshold
                )
                if check_label:
                    filter_result = (
                        filter_result and predicted_bbox.label_id == gt_bbox.label_id
                    )
                ious_single_gt_bool.append(filter_result)

            ious_matrix.append(ious_single_gt)
            ious_matrix_bool.append(ious_single_gt_bool)

        return ious_matrix_bool, ious_matrix

    @staticmethod
    def compute_metrics(ious_matrix_bool: List[List[bool]]) -> Tuple[int, int, int]:
        """
        Computes TP, FP, FN, TN from IoU match matrix.
        TP: A prediction correctly matches a ground truth (i.e. True in ious_matrix_bool).
        FP: A prediction doesn’t match any ground truth (i.e. all False in its column, column is all GTs).
        FN: A ground truth has no matching predictions (i.e. all False in its row, row is all predictions).

        Args:
            ious_matrix_bool: A 2D list of booleans indicating IoU threshold match.

        Returns:
            A tuple: (TP, FP, FN, TN)
        """
        num_gt = len(ious_matrix_bool)
        num_pred = len(ious_matrix_bool[0])

        matched_preds = [False] * num_pred
        matched_gts = [False] * num_gt

        for i, gt_row in enumerate(ious_matrix_bool):
            for j, matched in enumerate(gt_row):
                if matched and not matched_preds[j] and not matched_gts[i]:
                    matched_preds[j] = True
                    matched_gts[i] = True

        tp = sum(matched_gts)
        fn = matched_gts.count(False)
        fp = matched_preds.count(False)

        return tp, fp, fn

    @staticmethod
    def evaluate_dataset(
        dataset: UNDataset,
        predicted_dataset: UNDataset,
        iou_threshold_for_match: float = 0.5,
        score_threshold: float = 0.5,
        check_label: bool = False,
        return_global: bool = False,
    ) -> List[Tuple[int, int, int]]:
        # Tagged samples

        gt_keys = dataset.sample.keys()

        if return_global:
            tp, fp, fn = 0, 0, 0

        # Accumulate metrics (tp, fp, fn) for each ground truth sample for each predicted sample
        metrics_dataset = []
        for idx in gt_keys:
            ious_matrix_bool, _ = Evaluator.sample_iou(
                dataset.sample[idx],
                predicted_dataset[idx],
                iou_threshold_for_match,
                score_threshold,
                check_label,
            )
            local_tp, local_fp, local_fn = Evaluator.compute_metrics(ious_matrix_bool)
            metrics_dataset.append((local_tp, local_fp, local_fn))

            # Accumulate metrics
            if return_global:
                tp += local_tp
                fp += local_fp
                fn += local_fn

        if return_global:
            return metrics_dataset, (tp, fp, fn)
        return metrics_dataset


if __name__ == "__main__":

    def simple_test():
        # Define GT sample with one bounding box
        gt_bbox = UNBBox(coords=[10, 10, 50, 50], format="xyxy", label_id=1)
        gt_sample = UNSample(
            image_path="image1.jpg", image_w=100, image_h=100, bbox=[gt_bbox]
        )

        # Define predicted sample with one matching bounding box (IoU = 1.0, perfect match)
        pred_bbox = UNBBox(
            coords=[10, 10, 50, 50], format="xyxy", label_id=1, score=0.9
        )
        predicted_sample = UNSample(
            image_path="image1.jpg", image_w=100, image_h=100, bbox=[pred_bbox]
        )

        # Create dataset and add the ground truth sample
        dataset = UNDataset()
        dataset.add_sample(gt_sample)

        # Run evaluation
        results = Evaluator.evaluate_dataset(
            dataset,
            predicted_dataset=UNDataset(sample={0: predicted_sample}),
            iou_threshold_for_match=0.5,
            score_threshold=0.5,
            check_label=True,
        )

        # Expected metrics: 1 True Positive, 0 False Positives, 0 False Negatives
        expected = [(1, 0, 0)]
        print("Evaluation Results:", results)
        assert results == expected, f"Expected {expected}, but got {results}"

        print("Test passed: evaluate_dataset with perfect match.")

        # Add a negative test case with mismatched prediction
        pred_bbox_miss = UNBBox(
            coords=[60, 60, 80, 80], format="xyxy", label_id=1, score=0.9
        )
        predicted_sample_miss = UNSample(
            image_path="image1.jpg", image_w=100, image_h=100, bbox=[pred_bbox_miss]
        )

        # Run evaluation again
        results_miss = Evaluator.evaluate_dataset(
            dataset,
            predicted_dataset=UNDataset(sample={0: predicted_sample_miss}),
            iou_threshold_for_match=0.5,
            score_threshold=0.5,
            check_label=True,
        )

        # Expected: 0 TP, 1 FP, 1 FN (no matches)
        expected_miss = [(0, 1, 1)]
        print("Evaluation Results (mismatch):", results_miss)
        assert (
            results_miss == expected_miss
        ), f"Expected {expected_miss}, but got {results_miss}"

        print("Test passed: evaluate_dataset with no match.")

    simple_test()
