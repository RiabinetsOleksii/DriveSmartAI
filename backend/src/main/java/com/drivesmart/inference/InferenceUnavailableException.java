package com.drivesmart.inference;

public class InferenceUnavailableException extends RuntimeException {
    public InferenceUnavailableException(String message) {
        super(message);
    }

    public InferenceUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
