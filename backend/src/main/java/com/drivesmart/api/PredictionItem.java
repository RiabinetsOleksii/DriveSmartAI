package com.drivesmart.api;

public record PredictionItem(int classId, String label, double confidence) {}
