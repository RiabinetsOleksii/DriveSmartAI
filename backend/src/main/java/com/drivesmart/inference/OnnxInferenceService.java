package com.drivesmart.inference;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

@Service
public class OnnxInferenceService implements AutoCloseable {
    private final String modelPath;
    private final String expectedSha256;
    private final ObjectMapper objectMapper;
    private final Map<Integer, String> classMapping;
    private OrtEnvironment environment;
    private OrtSession session;
    private volatile String unavailableReason;

    public OnnxInferenceService(
            @Value("${onnx.model.path}") String modelPath,
            @Value("${onnx.model.sha256:}") String expectedSha256,
            ObjectMapper objectMapper
    ) {
        this.modelPath = modelPath;
        this.expectedSha256 = expectedSha256;
        this.objectMapper = objectMapper;
        this.classMapping = loadClassMapping();
    }

    @PostConstruct
    public void initialize() {
        try {
            Path path = resolveModelPath();
            if (!Files.isRegularFile(path)) {
                unavailableReason = "ONNX model is missing: " + path.toAbsolutePath();
                return;
            }
            verifySha256(path);
            environment = OrtEnvironment.getEnvironment();
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setInterOpNumThreads(1);
                options.setIntraOpNumThreads(1);
                session = environment.createSession(path.toString(), options);
            }
            unavailableReason = null;
        } catch (Exception e) {
            unavailableReason = "ONNX model could not be loaded: " + e.getMessage();
        }
    }

    public PredictionResult predict(float[] tensor) {
        ensureAvailable();
        synchronized (session) {
            try {
                FloatBuffer inputBuffer = FloatBuffer.wrap(tensor);
                try (OnnxTensor input = OnnxTensor.createTensor(environment, inputBuffer, new long[]{1, 3, 64, 64})) {
                    OrtSession.Result result = session.run(Map.of("input", input));
                    try (result) {
                        float[] logits = extractLogits(result);
                        if (logits.length != 43) {
                            throw new IllegalStateException("Expected 43 logits, received " + logits.length);
                        }
                        List<Double> probabilities = softmax(logits);
                        List<PredictionItem> top = new ArrayList<>();
                        for (int index = 0; index < probabilities.size(); index++) {
                            top.add(new PredictionItem(index, classMapping.get(index), probabilities.get(index)));
                        }
                        top.sort(Comparator.comparing(PredictionItem::confidence).reversed());
                        List<PredictionItem> topThree = top.subList(0, Math.min(3, top.size()));
                        PredictionItem best = topThree.getFirst();
                        return new PredictionResult(best.classId(), best.label(), best.confidence(), topThree);
                    }
                }
            } catch (Exception e) {
                throw new InferenceUnavailableException("ONNX inference failed", e);
            }
        }
    }

    private float[] extractLogits(OrtSession.Result result) throws Exception {
        Object value = result.get(0).getValue();
        if (value instanceof float[][] batch) {
            return batch[0];
        }
        if (value instanceof float[] vector) {
            return vector;
        }
        throw new IllegalStateException("Unexpected ONNX output type: " + value.getClass().getName());
    }

    static List<Double> softmax(float[] logits) {
        double max = logits[0];
        for (float logit : logits) {
            max = Math.max(max, logit);
        }
        double sum = 0.0;
        double[] exponentials = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exponentials[i] = Math.exp(logits[i] - max);
            sum += exponentials[i];
        }
        List<Double> probabilities = new ArrayList<>(logits.length);
        for (double exponential : exponentials) {
            probabilities.add(exponential / sum);
        }
        return probabilities;
    }

    private void ensureAvailable() {
        if (session == null) {
            throw new InferenceUnavailableException(
                    unavailableReason != null ? unavailableReason : "ONNX model is not available"
            );
        }
    }

    private Path resolveModelPath() {
        Path path = Path.of(modelPath);
        if (path.isAbsolute()) {
            return path.normalize();
        }
        Path workingDirectory = Path.of("").toAbsolutePath().normalize();
        Path resolved = workingDirectory.resolve(path).normalize();
        if (Files.isRegularFile(resolved)) {
            return resolved;
        }
        return workingDirectory.resolve("../").resolve(path).normalize();
    }

    private void verifySha256(Path path) throws Exception {
        if (expectedSha256 == null || expectedSha256.isBlank()) {
            return;
        }
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] bytes = Files.readAllBytes(path);
        String actual = bytesToHex(digest.digest(bytes));
        if (!actual.equalsIgnoreCase(expectedSha256.trim())) {
            throw new IllegalStateException("ONNX SHA-256 mismatch");
        }
    }

    private Map<Integer, String> loadClassMapping() {
        try {
            ClassPathResource resource = new ClassPathResource("class_mapping.json");
            Map<String, String> mapping = objectMapper.readValue(
                    resource.getInputStream(),
                    new TypeReference<Map<String, String>>() {}
            );
            if (mapping.size() != 43) {
                throw new IllegalStateException("Class mapping must contain exactly 43 entries");
            }
            return mapping.entrySet().stream()
                    .collect(java.util.stream.Collectors.toMap(
                            entry -> Integer.parseInt(entry.getKey()),
                            Map.Entry::getValue
                    ));
        } catch (Exception e) {
            throw new IllegalStateException("Could not load class mapping", e);
        }
    }

    private static String bytesToHex(byte[] bytes) {
        StringBuilder value = new StringBuilder(bytes.length * 2);
        for (byte item : bytes) {
            value.append(Character.forDigit((item >>> 4) & 0xf, 16));
            value.append(Character.forDigit(item & 0xf, 16));
        }
        return value.toString();
    }

    @Override
    @PreDestroy
    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (Exception ignored) {
                // Spring shutdown is not an inference failure.
            }
            session = null;
        }
        if (environment != null) {
            try {
                environment.close();
            } catch (Exception ignored) {
                // Spring shutdown is not an inference failure.
            }
            environment = null;
        }
    }

    public record PredictionItem(int classId, String label, double confidence) {}
    public record PredictionResult(int classId, String label, double confidence, List<PredictionItem> topPredictions) {}
}
