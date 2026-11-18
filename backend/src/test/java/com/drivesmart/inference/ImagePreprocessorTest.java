package com.drivesmart.inference;

import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.InputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ImagePreprocessorTest {
    @Test
    void solidRgbImageHasExpectedShapeChannelsAndNormalization() throws Exception {
        BufferedImage image = new BufferedImage(23, 17, BufferedImage.TYPE_3BYTE_BGR);
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                image.setRGB(x, y, 0xff646e78);
            }
        }
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        ImageIO.write(image, "png", bytes);

        float[] tensor = new ImagePreprocessor().preprocess(bytes.toByteArray());

        assertEquals(3 * 64 * 64, tensor.length);
        assertEquals((100f / 255f - 0.485f) / 0.229f, tensor[0], 1e-5f);
        assertEquals((110f / 255f - 0.456f) / 0.224f, tensor[64 * 64], 1e-5f);
        assertEquals((120f / 255f - 0.406f) / 0.225f, tensor[2 * 64 * 64], 1e-5f);
        for (float value : tensor) {
            assertTrue(Float.isFinite(value));
        }
    }

    @Test
    void preprocessingMatchesPythonFixtureWithinTolerance() throws Exception {
        try (InputStream imageStream = getClass().getResourceAsStream("/preprocessing-parity.png");
             InputStream tensorStream = getClass().getResourceAsStream("/preprocessing-parity.json")) {
            if (imageStream == null || tensorStream == null) {
                throw new AssertionError("Python preprocessing fixture is missing");
            }
            float[] actual = new ImagePreprocessor().preprocess(imageStream.readAllBytes());
            float[][][] expected = new ObjectMapper().readValue(tensorStream, float[][][].class);
            float maxDifference = 0.0f;
            for (int channel = 0; channel < 3; channel++) {
                for (int y = 0; y < 64; y++) {
                    for (int x = 0; x < 64; x++) {
                        int index = channel * 64 * 64 + y * 64 + x;
                        maxDifference = Math.max(maxDifference, Math.abs(actual[index] - expected[channel][y][x]));
                    }
                }
            }
            assertTrue(maxDifference < 0.02f, "Java/Python max preprocessing difference was " + maxDifference);
        }
    }
}
