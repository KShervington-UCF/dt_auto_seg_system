import React from 'react';
import { EngineStatus } from '../types';
import { CheckCircle, Circle, Loader2, XCircle } from 'lucide-react';

interface ProcessingStatusProps {
  engines: EngineStatus[];
}

export function ProcessingStatus({ engines }: ProcessingStatusProps) {
  const getStatusIcon = (status: EngineStatus['status']) => {
    switch (status) {
      case 'complete':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'processing':
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Circle className="w-5 h-5 text-gray-300" />;
    }
  };

  return (
    <div className="space-y-3">
      {engines.map((engine) => (
        <div key={engine.name} className="flex items-center space-x-3">
          {getStatusIcon(engine.status)}
          <span className="text-sm font-medium text-gray-700">{engine.name}</span>
        </div>
      ))}
    </div>
  );
}