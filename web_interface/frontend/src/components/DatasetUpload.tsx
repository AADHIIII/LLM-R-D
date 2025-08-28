import React, { useState, useRef } from 'react';
import { CloudArrowUpIcon, DocumentTextIcon, XMarkIcon } from '@heroicons/react/24/outline';
import Button from './Button';
import Card from './Card';

interface DatasetUploadProps {
  onUpload: (file: File) => void;
  loading?: boolean;
  accept?: string;
}

interface UploadProgress {
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  message?: string;
}

const DatasetUpload: React.FC<DatasetUploadProps> = ({
  onUpload,
  loading = false,
  accept = '.jsonl,.csv',
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (file: File) => {
    // Validate file type
    const allowedTypes = ['.jsonl', '.csv'];
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
      setUploadProgress({
        progress: 0,
        status: 'error',
        message: 'Please select a JSONL or CSV file',
      });
      return;
    }

    // Validate file size (max 100MB)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
      setUploadProgress({
        progress: 0,
        status: 'error',
        message: 'File size must be less than 100MB',
      });
      return;
    }

    setSelectedFile(file);
    setUploadProgress(null);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleUpload = () => {
    if (!selectedFile) return;

    // Simulate upload progress
    setUploadProgress({ progress: 0, status: 'uploading' });
    
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (!prev) return null;
        
        const newProgress = prev.progress + 10;
        
        if (newProgress >= 100) {
          clearInterval(progressInterval);
          return { progress: 100, status: 'processing', message: 'Processing dataset...' };
        }
        
        return { ...prev, progress: newProgress };
      });
    }, 200);

    // Simulate processing
    setTimeout(() => {
      setUploadProgress({ progress: 100, status: 'completed', message: 'Dataset uploaded successfully!' });
      onUpload(selectedFile);
    }, 3000);
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setUploadProgress(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Card title="Dataset Upload" subtitle="Upload your training dataset in JSONL or CSV format">
      <div className="space-y-4">
        {!selectedFile ? (
          <div
            className={`relative border-2 border-dashed rounded-lg p-6 transition-colors ${
              dragActive
                ? 'border-primary-400 bg-primary-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="text-center">
              <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
              <div className="mt-4">
                <label htmlFor="file-upload" className="cursor-pointer">
                  <span className="mt-2 block text-sm font-medium text-gray-900">
                    Drop your dataset here, or{' '}
                    <span className="text-primary-600 hover:text-primary-500">browse</span>
                  </span>
                  <input
                    ref={fileInputRef}
                    id="file-upload"
                    name="file-upload"
                    type="file"
                    className="sr-only"
                    accept={accept}
                    onChange={handleFileInputChange}
                  />
                </label>
                <p className="mt-1 text-xs text-gray-500">
                  JSONL or CSV files up to 100MB
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <DocumentTextIcon className="h-8 w-8 text-gray-400" />
                <div>
                  <p className="text-sm font-medium text-gray-900">{selectedFile.name}</p>
                  <p className="text-xs text-gray-500">{formatFileSize(selectedFile.size)}</p>
                </div>
              </div>
              {!uploadProgress && (
                <button
                  onClick={handleRemoveFile}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <XMarkIcon className="h-5 w-5" />
                </button>
              )}
            </div>

            {uploadProgress && (
              <div className="mt-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">
                    {uploadProgress.status === 'uploading' && 'Uploading...'}
                    {uploadProgress.status === 'processing' && 'Processing...'}
                    {uploadProgress.status === 'completed' && 'Completed'}
                    {uploadProgress.status === 'error' && 'Error'}
                  </span>
                  <span className="text-gray-600">{uploadProgress.progress}%</span>
                </div>
                <div className="mt-1 w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${
                      uploadProgress.status === 'error'
                        ? 'bg-red-500'
                        : uploadProgress.status === 'completed'
                        ? 'bg-green-500'
                        : 'bg-primary-500'
                    }`}
                    style={{ width: `${uploadProgress.progress}%` }}
                  />
                </div>
                {uploadProgress.message && (
                  <p className={`mt-2 text-sm ${
                    uploadProgress.status === 'error' ? 'text-red-600' : 'text-gray-600'
                  }`}>
                    {uploadProgress.message}
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {selectedFile && !uploadProgress && (
          <div className="flex justify-end">
            <Button onClick={handleUpload} loading={loading}>
              Upload Dataset
            </Button>
          </div>
        )}

        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Dataset Format Requirements</h4>
          <div className="text-xs text-gray-600 space-y-1">
            <p><strong>JSONL:</strong> Each line should contain a JSON object with "prompt" and "response" fields</p>
            <p><strong>CSV:</strong> Should have "prompt" and "response" columns</p>
            <p><strong>Example:</strong></p>
            <pre className="mt-2 bg-white p-2 rounded border text-xs">
{`{"prompt": "What is AI?", "response": "AI is artificial intelligence..."}`}
            </pre>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default DatasetUpload;