package com.drivesmart.inference;

import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

@Component
public class ImagePreprocessor {
    public static final int IMAGE_SIZE = 64;
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    public float[] preprocess(byte[] bytes) {
        if (bytes == null || bytes.length == 0) {
            throw new IllegalArgumentException("Uploaded file is empty");
        }
        try {
            BufferedImage source = ImageIO.read(new ByteArrayInputStream(bytes));
            if (source == null) {
                throw new IllegalArgumentException("The file is not a decodable PNG or JPEG image");
            }
            BufferedImage rgb = new BufferedImage(source.getWidth(), source.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
            Graphics2D graphics = rgb.createGraphics();
            try {
                graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                graphics.drawImage(source, 0, 0, null);
            } finally {
                graphics.dispose();
            }
            BufferedImage resized = resizeBilinear(rgb);
            return toNchw(resized);
        } catch (IOException e) {
            throw new IllegalArgumentException("The file is not a valid PNG or JPEG image", e);
        }
    }

    private float[] toNchw(BufferedImage image) {
        float[] tensor = new float[3 * IMAGE_SIZE * IMAGE_SIZE];
        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                int rgb = image.getRGB(x, y);
                int red = (rgb >>> 16) & 0xff;
                int green = (rgb >>> 8) & 0xff;
                int blue = rgb & 0xff;
                int offset = y * IMAGE_SIZE + x;
                tensor[offset] = (red / 255.0f - MEAN[0]) / STD[0];
                tensor[IMAGE_SIZE * IMAGE_SIZE + offset] = (green / 255.0f - MEAN[1]) / STD[1];
                tensor[2 * IMAGE_SIZE * IMAGE_SIZE + offset] = (blue / 255.0f - MEAN[2]) / STD[2];
            }
        }
        return tensor;
    }

    private BufferedImage resizeBilinear(BufferedImage source) {
        BufferedImage resized = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_3BYTE_BGR);
        double scaleX = (double) source.getWidth() / IMAGE_SIZE;
        double scaleY = (double) source.getHeight() / IMAGE_SIZE;
        for (int y = 0; y < IMAGE_SIZE; y++) {
            double sourceY = (y + 0.5d) * scaleY - 0.5d;
            int y0 = (int) Math.floor(sourceY);
            int y1 = y0 + 1;
            double fy = sourceY - y0;
            y0 = Math.max(0, Math.min(source.getHeight() - 1, y0));
            y1 = Math.max(0, Math.min(source.getHeight() - 1, y1));
            for (int x = 0; x < IMAGE_SIZE; x++) {
                double sourceX = (x + 0.5d) * scaleX - 0.5d;
                int x0 = (int) Math.floor(sourceX);
                int x1 = x0 + 1;
                double fx = sourceX - x0;
                x0 = Math.max(0, Math.min(source.getWidth() - 1, x0));
                x1 = Math.max(0, Math.min(source.getWidth() - 1, x1));

                int rgb00 = source.getRGB(x0, y0);
                int rgb10 = source.getRGB(x1, y0);
                int rgb01 = source.getRGB(x0, y1);
                int rgb11 = source.getRGB(x1, y1);
                int red = bilinearChannel(rgb00, rgb10, rgb01, rgb11, 16, fx, fy);
                int green = bilinearChannel(rgb00, rgb10, rgb01, rgb11, 8, fx, fy);
                int blue = bilinearChannel(rgb00, rgb10, rgb01, rgb11, 0, fx, fy);
                int rgb = (red << 16) | (green << 8) | blue;
                resized.setRGB(x, y, rgb);
            }
        }
        return resized;
    }

    private int bilinearChannel(int rgb00, int rgb10, int rgb01, int rgb11, int shift, double fx, double fy) {
        double c00 = (rgb00 >>> shift) & 0xff;
        double c10 = (rgb10 >>> shift) & 0xff;
        double c01 = (rgb01 >>> shift) & 0xff;
        double c11 = (rgb11 >>> shift) & 0xff;
        double top = c00 + (c10 - c00) * fx;
        double bottom = c01 + (c11 - c01) * fx;
        return (int) Math.round(top + (bottom - top) * fy);
    }
}
