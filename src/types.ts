export interface ProcessedImage {
  id: string;
  url: string;
  annotations: any[];
  classification: string;
}

export interface ProcessingResult {
  images: ProcessedImage[];
  segmentationData: any;
  classificationData: any;
}

export interface EngineStatus {
  name: string;
  status: 'idle' | 'processing' | 'complete' | 'error';
}