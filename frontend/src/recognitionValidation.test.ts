import assert from 'node:assert/strict'
import test from 'node:test'
import { MAX_IMAGE_SIZE, validateImageFile } from './recognitionValidation.ts'

test('accepts JPEG and PNG files within the 5 MB limit', () => {
  assert.equal(validateImageFile({ type: 'image/jpeg', size: 1024 } as File), null)
  assert.equal(validateImageFile({ type: 'image/png', size: MAX_IMAGE_SIZE } as File), null)
})

test('rejects unsupported formats, empty files and oversized files', () => {
  assert.match(validateImageFile({ type: 'image/webp', size: 1024 } as File) ?? '', /JPEG або PNG/)
  assert.match(validateImageFile({ type: 'image/png', size: 0 } as File) ?? '', /порожнє/)
  assert.match(validateImageFile({ type: 'image/png', size: MAX_IMAGE_SIZE + 1 } as File) ?? '', /5 МБ/)
})
