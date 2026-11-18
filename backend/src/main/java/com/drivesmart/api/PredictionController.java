package com.drivesmart.api;

import com.drivesmart.service.PredictionService;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
public class PredictionController {
    private static final long MAX_FILE_SIZE = 5L * 1024L * 1024L;

    private final PredictionService predictionService;

    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @PostMapping(value = "/api/predict", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public PredictionResponse predict(@RequestParam("file") MultipartFile file) throws IOException {
        if (file.isEmpty()) {
            throw new IllegalArgumentException("Uploaded file is empty");
        }
        if (file.getSize() > MAX_FILE_SIZE) {
            throw new IllegalArgumentException("Uploaded file exceeds 5 MB");
        }
        byte[] bytes = file.getBytes();
        if (!isPngOrJpeg(bytes)) {
            throw new IllegalArgumentException("Only PNG and JPEG images are supported");
        }
        return predictionService.predict(bytes);
    }

    private boolean isPngOrJpeg(byte[] bytes) {
        if (bytes.length >= 8
                && (bytes[0] & 0xff) == 0x89
                && (bytes[1] & 0xff) == 0x50
                && (bytes[2] & 0xff) == 0x4e
                && (bytes[3] & 0xff) == 0x47
                && (bytes[4] & 0xff) == 0x0d
                && (bytes[5] & 0xff) == 0x0a
                && (bytes[6] & 0xff) == 0x1a
                && (bytes[7] & 0xff) == 0x0a) {
            return true;
        }
        return bytes.length >= 3
                && (bytes[0] & 0xff) == 0xff
                && (bytes[1] & 0xff) == 0xd8
                && (bytes[2] & 0xff) == 0xff;
    }
}
