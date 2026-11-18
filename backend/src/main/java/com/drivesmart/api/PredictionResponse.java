package com.drivesmart.api;

import java.util.List;

public record PredictionResponse(
        int classId,
        String label,
        double confidence,
        List<PredictionItem> topPredictions
) {}
