import React, { useState } from 'react';
import { ImageUploader } from './components/ImageUploader';
import { ProcessingStatus } from './components/ProcessingStatus';
import { ResultViewer } from './components/ResultViewer';
import { Brain } from 'lucide-react';
import type { EngineStatus, ProcessedImage } from './types';
import axios from 'axios';

function App() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [engines, setEngines] = useState<EngineStatus[]>([
    { name: 'Image Processing', status: 'idle' },
    { name: 'Segmentation', status: 'idle' },
    { name: 'Classification', status: 'idle' },
    { name: 'Integration', status: 'idle' },
    { name: 'Output', status: 'idle' }
  ]);
  const [results, setResults] = useState<{
    images: ProcessedImage[];
    segmentationData: any;
    classificationData: any;
  } | null>(null);

  const handleUpload = async (files: File[]) => {
    setIsProcessing(true);
    const formData = new FormData();
    files.forEach((file) => formData.append('images', file));

    try {
      // Simulate processing with engines
      for (let i = 0; i < engines.length; i++) {
        setEngines((prev) =>
          prev.map((engine, index) => ({
            ...engine,
            status: index === i ? 'processing' : engine.status
          }))
        );

        // Simulate processing time
        await new Promise((resolve) => setTimeout(resolve, 1500));

        setEngines((prev) =>
          prev.map((engine, index) => ({
            ...engine,
            status: index === i ? 'complete' : engine.status
          }))
        );
      }

      // Simulate API response
      const mockResults = {
        images: files.map((file, index) => ({
          id: `img-${index}`,
          url: URL.createObjectURL(file),
          annotations: [
            { type: 'bbox', coordinates: [100, 100, 200, 200] },
            { type: 'polygon', points: [[150, 150], [160, 160], [170, 150]] }
          ],
          classification: ['object-' + index]
        })),
        segmentationData: {
          version: '1.0',
          timestamp: new Date().toISOString(),
          segments: files.map((_, i) => ({
            imageId: `img-${i}`,
            regions: [
              { id: i, label: 'region-1', confidence: 0.95 },
              { id: i + 1, label: 'region-2', confidence: 0.87 }
            ]
          }))
        },
        classificationData: {
          version: '1.0',
          timestamp: new Date().toISOString(),
          classifications: files.map((_, i) => ({
            imageId: `img-${i}`,
            predictions: [
              { label: 'class-A', confidence: 0.92 },
              { label: 'class-B', confidence: 0.78 }
            ]
          }))
        }
      };

      setResults(mockResults);
    } catch (error) {
      console.error('Processing failed:', error);
      setEngines((prev) =>
        prev.map((engine) => ({
          ...engine,
          status: engine.status === 'processing' ? 'error' : engine.status
        }))
      );
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <Brain className="h-12 w-12 text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Image Processing System
          </h1>
          <p className="text-lg text-gray-600">
            Upload images for processing, segmentation, and classification
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">
                Upload Images
              </h2>
              <ImageUploader onUpload={handleUpload} isProcessing={isProcessing} />
            </div>

            {results && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">
                  Processing Results
                </h2>
                <ResultViewer
                  images={results.images}
                  segmentationData={results.segmentationData}
                  classificationData={results.classificationData}
                />
              </div>
            )}
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">
              Processing Status
            </h2>
            <ProcessingStatus engines={engines} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;