package com.drivesmart.service;

import com.drivesmart.api.PredictionItem;
import com.drivesmart.api.PredictionResponse;
import com.drivesmart.inference.ImagePreprocessor;
import com.drivesmart.inference.OnnxInferenceService;
import org.springframework.stereotype.Service;

@Service
public class PredictionService {
    private final ImagePreprocessor imagePreprocessor;
    private final OnnxInferenceService inferenceService;

    public PredictionService(ImagePreprocessor imagePreprocessor, OnnxInferenceService inferenceService) {
        this.imagePreprocessor = imagePreprocessor;
        this.inferenceService = inferenceService;
    }

    public PredictionResponse predict(byte[] imageBytes) {
        float[] tensor = imagePreprocessor.preprocess(imageBytes);
        OnnxInferenceService.PredictionResult result = inferenceService.predict(tensor);
        return new PredictionResponse(
                result.classId(),
                result.label(),
                result.confidence(),
                result.topPredictions().stream()
                        .map(item -> new PredictionItem(item.classId(), item.label(), item.confidence()))
                        .toList()
        );
    }
}
