package com.drivesmart.inference;

import com.drivesmart.api.PredictionResponse;
import com.drivesmart.service.PredictionService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

@SpringBootTest
class OnnxRealInferenceTest {
    @Autowired
    private PredictionService predictionService;

    @Test
    void realGtsrbImageProducesPrediction() throws Exception {
        assumeTrue(Files.isRegularFile(Path.of("../ml-training/outputs/custom-cnn/best_cnn.onnx")),
                "Local ONNX model is not available");
        try (InputStream stream = getClass().getResourceAsStream("/gtsrb-00000.png")) {
            if (stream == null) {
                throw new AssertionError("GTSRB test fixture is missing");
            }
            PredictionResponse response = predictionService.predict(stream.readAllBytes());
            assertEquals(3, response.topPredictions().size());
            assertEquals(16, response.classId(), "Ground-truth class ID is 16 for GTSRB 00000");
        }
    }
}
