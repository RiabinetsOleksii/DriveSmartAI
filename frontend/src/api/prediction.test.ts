import assert from 'node:assert/strict'
import test from 'node:test'
import { isPredictionResponse } from './prediction.ts'

test('accepts the Java PredictionResponse shape', () => {
  const response = {
    classId: 16,
    label: 'Vehicles over 3.5 metric tons prohibited',
    confidence: 0.9975,
    topPredictions: [
      { classId: 16, label: 'Vehicles over 3.5 metric tons prohibited', confidence: 0.9975 },
      { classId: 15, label: 'No vehicles', confidence: 0.0012 },
    ],
  }

  assert.equal(isPredictionResponse(response), true)
})

test('rejects malformed prediction responses', () => {
  assert.equal(isPredictionResponse({ classId: 16, label: 'Test', confidence: 0.5 }), false)
  assert.equal(isPredictionResponse({
    classId: '16',
    label: 'Test',
    confidence: 0.5,
    topPredictions: [],
  }), false)
})
