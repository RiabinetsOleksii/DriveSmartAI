package com.drivesmart.api;

import com.drivesmart.inference.InferenceUnavailableException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

import java.util.Map;

@RestControllerAdvice
public class PredictionExceptionHandler {
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, Object>> badRequest(IllegalArgumentException exception) {
        return ResponseEntity.badRequest().body(Map.of(
                "error", "INVALID_IMAGE",
                "message", exception.getMessage()
        ));
    }

    @ExceptionHandler(InferenceUnavailableException.class)
    public ResponseEntity<Map<String, Object>> unavailable(InferenceUnavailableException exception) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(Map.of(
                "error", "MODEL_UNAVAILABLE",
                "message", exception.getMessage()
        ));
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<Map<String, Object>> tooLarge(MaxUploadSizeExceededException exception) {
        return ResponseEntity.badRequest().body(Map.of(
                "error", "INVALID_IMAGE",
                "message", "Uploaded file exceeds 5 MB"
        ));
    }

    @ExceptionHandler(MissingServletRequestParameterException.class)
    public ResponseEntity<Map<String, Object>> missingFile(MissingServletRequestParameterException exception) {
        return ResponseEntity.badRequest().body(Map.of(
                "error", "INVALID_IMAGE",
                "message", "Multipart field 'file' is required"
        ));
    }
}
