package com.drivesmart.inference;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OnnxInferenceServiceTest {
    @Test
    void softmaxProducesStableProbabilitiesAndTopThreeCanBeOrdered() {
        float[] logits = new float[43];
        logits[14] = 4.0f;
        logits[13] = 2.0f;
        logits[12] = 1.0f;

        List<Double> probabilities = OnnxInferenceService.softmax(logits);
        double sum = probabilities.stream().mapToDouble(Double::doubleValue).sum();

        assertEquals(1.0d, sum, 1e-9);
        assertTrue(probabilities.get(14) > probabilities.get(13));
        assertTrue(probabilities.get(13) > probabilities.get(12));
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < probabilities.size(); i++) {
            indices.add(i);
        }
        indices.sort((left, right) -> Double.compare(probabilities.get(right), probabilities.get(left)));
        assertEquals(List.of(14, 13, 12), indices.subList(0, 3));
    }
}
