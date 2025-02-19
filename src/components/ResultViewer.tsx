import React, { useState } from 'react';
import { ProcessedImage } from '../types';
import { Download, ChevronLeft, ChevronRight } from 'lucide-react';

interface ResultViewerProps {
  images: ProcessedImage[];
  segmentationData: any;
  classificationData: any;
}

export function ResultViewer({ images, segmentationData, classificationData }: ResultViewerProps) {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [activeTab, setActiveTab] = useState<'segmentation' | 'classification'>('segmentation');

  const handlePrevious = () => {
    setCurrentImageIndex((prev) => (prev > 0 ? prev - 1 : images.length - 1));
  };

  const handleNext = () => {
    setCurrentImageIndex((prev) => (prev < images.length - 1 ? prev + 1 : 0));
  };

  const downloadJson = (data: any, filename: string) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden">
        {images.length > 0 && (
          <img
            src={images[currentImageIndex].url}
            alt={`Processed image ${currentImageIndex + 1}`}
            className="w-full h-full object-contain"
          />
        )}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center space-x-4 bg-black/50 rounded-full px-4 py-2">
          <button
            onClick={handlePrevious}
            className="text-white hover:text-blue-200 transition-colors"
          >
            <ChevronLeft className="w-6 h-6" />
          </button>
          <span className="text-white text-sm">
            {currentImageIndex + 1} / {images.length}
          </span>
          <button
            onClick={handleNext}
            className="text-white hover:text-blue-200 transition-colors"
          >
            <ChevronRight className="w-6 h-6" />
          </button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="border-b">
          <nav className="flex">
            <button
              className={`px-4 py-2 text-sm font-medium ${
                activeTab === 'segmentation'
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab('segmentation')}
            >
              Segmentation Data
            </button>
            <button
              className={`px-4 py-2 text-sm font-medium ${
                activeTab === 'classification'
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab('classification')}
            >
              Classification Data
            </button>
          </nav>
        </div>

        <div className="p-4">
          <pre className="bg-gray-50 rounded p-4 text-sm overflow-auto max-h-60">
            {JSON.stringify(
              activeTab === 'segmentation' ? segmentationData : classificationData,
              null,
              2
            )}
          </pre>
          <button
            onClick={() =>
              downloadJson(
                activeTab === 'segmentation' ? segmentationData : classificationData,
                `${activeTab}-data.json`
              )
            }
            className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <Download className="w-4 h-4 mr-2" />
            Download {activeTab === 'segmentation' ? 'Segmentation' : 'Classification'} Data
          </button>
        </div>
      </div>
    </div>
  );
}