package com.drivesmart.api;

import com.drivesmart.inference.InferenceUnavailableException;
import com.drivesmart.service.PredictionService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.List;
import org.springframework.mock.web.MockMultipartFile;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(PredictionController.class)
class PredictionControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private PredictionService predictionService;

    @Test
    void predictReturnsTopPrediction() throws Exception {
        when(predictionService.predict(any(byte[].class))).thenReturn(new PredictionResponse(
                14,
                "Stop",
                0.9821,
                List.of(
                        new PredictionItem(14, "Stop", 0.9821),
                        new PredictionItem(13, "Yield", 0.0102),
                        new PredictionItem(12, "Priority road", 0.0035)
                )
        ));
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(64, 64, BufferedImage.TYPE_3BYTE_BGR), "png", bytes);

        mockMvc.perform(multipart("/api/predict").file(
                        new MockMultipartFile("file", "sign.png", "image/png", bytes.toByteArray())
                ))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.classId").value(14))
                .andExpect(jsonPath("$.label").value("Stop"))
                .andExpect(jsonPath("$.topPredictions.length()").value(3));
    }

    @Test
    void emptyFileIsRejected() throws Exception {
        mockMvc.perform(multipart("/api/predict").file(
                        new MockMultipartFile("file", "empty.png", "image/png", new byte[0])
                ))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("INVALID_IMAGE"));
    }

    @Test
    void unsupportedFormatIsRejected() throws Exception {
        mockMvc.perform(multipart("/api/predict").file(
                        new MockMultipartFile("file", "sign.txt", "text/plain", "not an image".getBytes())
                ))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("INVALID_IMAGE"));
    }

    @Test
    void unavailableModelIsReportedWithoutStackTrace() throws Exception {
        when(predictionService.predict(any(byte[].class)))
                .thenThrow(new InferenceUnavailableException("ONNX model is missing"));
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(64, 64, BufferedImage.TYPE_3BYTE_BGR), "png", bytes);

        mockMvc.perform(multipart("/api/predict").file(
                        new MockMultipartFile("file", "sign.png", "image/png", bytes.toByteArray())
                ))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.error").value("MODEL_UNAVAILABLE"))
                .andExpect(jsonPath("$.stackTrace").doesNotExist());
    }
}
