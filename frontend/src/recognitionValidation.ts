export const MAX_IMAGE_SIZE = 5 * 1024 * 1024

export function validateImageFile(file: File | undefined): string | null {
  if (!file) return 'Будь ласка, оберіть фотографію.'
  if (!['image/jpeg', 'image/png'].includes(file.type)) {
    return 'Підтримуються лише зображення JPEG або PNG.'
  }
  if (file.size === 0) {
    return 'Вибране зображення порожнє.'
  }
  if (file.size > MAX_IMAGE_SIZE) {
    return 'Розмір зображення не може перевищувати 5 МБ.'
  }
  return null
}

export function formatFileSize(size: number): string {
  if (size < 1024) return `${size} Б`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} КБ`
  return `${(size / (1024 * 1024)).toFixed(1)} МБ`
}
