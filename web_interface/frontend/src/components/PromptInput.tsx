import React, { useState, useRef, useEffect } from 'react';
import { PlusIcon, XMarkIcon, EyeIcon, EyeSlashIcon } from '@heroicons/react/24/outline';
import Button from './Button';
import Card from './Card';

interface PromptInputProps {
  prompts: string[];
  onPromptsChange: (prompts: string[]) => void;
  maxPrompts?: number;
}

interface HighlightedTextProps {
  text: string;
  className?: string;
}

// Syntax highlighting component for prompts
const HighlightedText: React.FC<HighlightedTextProps> = ({ text, className = '' }) => {
  const highlightSyntax = (text: string) => {
    // Define patterns for common prompt elements
    const patterns = [
      { regex: /\{[^}]+\}/g, className: 'text-blue-600 font-semibold' }, // Variables {variable}
      { regex: /\[[^\]]+\]/g, className: 'text-green-600 font-semibold' }, // Instructions [instruction]
      { regex: /"""[^"]*"""/g, className: 'text-purple-600 bg-purple-50' }, // Triple quotes
      { regex: /```[^`]*```/g, className: 'text-gray-800 bg-gray-100 font-mono' }, // Code blocks
      { regex: /\*\*[^*]+\*\*/g, className: 'font-bold text-gray-900' }, // Bold text
      { regex: /\*[^*]+\*/g, className: 'italic text-gray-700' }, // Italic text
      { regex: /^(System:|User:|Assistant:|Human:)/gm, className: 'text-red-600 font-bold' }, // Role indicators
    ];

    let highlightedText = text;
    let parts: Array<{ text: string; className?: string }> = [{ text: highlightedText }];

    patterns.forEach(pattern => {
      const newParts: Array<{ text: string; className?: string }> = [];
      
      parts.forEach(part => {
        if (part.className) {
          // Already highlighted, don't process further
          newParts.push(part);
          return;
        }

        const matches = Array.from(part.text.matchAll(pattern.regex));
        if (matches.length === 0) {
          newParts.push(part);
          return;
        }

        let lastIndex = 0;
        matches.forEach(match => {
          // Add text before match
          if (match.index! > lastIndex) {
            newParts.push({ text: part.text.slice(lastIndex, match.index) });
          }
          // Add highlighted match
          newParts.push({ text: match[0], className: pattern.className });
          lastIndex = match.index! + match[0].length;
        });

        // Add remaining text
        if (lastIndex < part.text.length) {
          newParts.push({ text: part.text.slice(lastIndex) });
        }
      });

      parts = newParts;
    });

    return parts;
  };

  const parts = highlightSyntax(text);

  return (
    <div className={`whitespace-pre-wrap ${className}`}>
      {parts.map((part, index) => (
        <span key={index} className={part.className || ''}>
          {part.text}
        </span>
      ))}
    </div>
  );
};

const PromptInput: React.FC<PromptInputProps> = ({
  prompts,
  onPromptsChange,
  maxPrompts = 5,
}) => {
  const [newPrompt, setNewPrompt] = useState('');
  const [showPreview, setShowPreview] = useState<{ [key: number]: boolean }>({});

  const addPrompt = () => {
    if (newPrompt.trim() && prompts.length < maxPrompts) {
      onPromptsChange([...prompts, newPrompt.trim()]);
      setNewPrompt('');
    }
  };

  const removePrompt = (index: number) => {
    onPromptsChange(prompts.filter((_, i) => i !== index));
  };

  const updatePrompt = (index: number, value: string) => {
    const updatedPrompts = [...prompts];
    updatedPrompts[index] = value;
    onPromptsChange(updatedPrompts);
  };

  const togglePreview = (index: number) => {
    setShowPreview(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      addPrompt();
    }
  };

  return (
    <Card title="Prompt Variants" subtitle="Add multiple prompt variations to test and compare">
      <div className="space-y-4">
        {/* Existing prompts */}
        {prompts.map((prompt, index) => (
          <div key={index} className="relative">
            <div className="flex items-start justify-between mb-2">
              <label className="block text-sm font-medium text-gray-700">
                Prompt {index + 1}
              </label>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => togglePreview(index)}
                  className="text-gray-400 hover:text-blue-500 transition-colors"
                  type="button"
                  title={showPreview[index] ? "Hide preview" : "Show preview"}
                >
                  {showPreview[index] ? (
                    <EyeSlashIcon className="h-4 w-4" />
                  ) : (
                    <EyeIcon className="h-4 w-4" />
                  )}
                </button>
                <button
                  onClick={() => removePrompt(index)}
                  className="text-gray-400 hover:text-red-500 transition-colors"
                  type="button"
                >
                  <XMarkIcon className="h-4 w-4" />
                </button>
              </div>
            </div>
            
            {showPreview[index] ? (
              <div className="border border-gray-300 rounded-md p-3 bg-gray-50 min-h-[100px]">
                <HighlightedText 
                  text={prompt || "Enter your prompt here..."} 
                  className="text-sm"
                />
              </div>
            ) : (
              <textarea
                value={prompt}
                onChange={(e) => updatePrompt(index, e.target.value)}
                className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500 sm:text-sm font-mono"
                rows={4}
                placeholder="Enter your prompt here..."
              />
            )}
            
            <div className="mt-1 flex items-center justify-between text-xs text-gray-500">
              <span>{prompt.length} characters</span>
              {showPreview[index] && (
                <span className="text-blue-600">Preview mode - click eye icon to edit</span>
              )}
            </div>
          </div>
        ))}

        {/* Add new prompt */}
        {prompts.length < maxPrompts && (
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-4">
            <div className="mb-2">
              <label className="block text-sm font-medium text-gray-700">
                New Prompt {prompts.length + 1}
              </label>
            </div>
            <textarea
              value={newPrompt}
              onChange={(e) => setNewPrompt(e.target.value)}
              onKeyDown={handleKeyPress}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500 sm:text-sm font-mono"
              rows={4}
              placeholder="Enter your prompt here... (Ctrl/Cmd + Enter to add)"
            />
            <div className="mt-2 flex items-center justify-between">
              <div className="text-xs text-gray-500">
                {newPrompt.length} characters
              </div>
              <Button
                size="sm"
                onClick={addPrompt}
                disabled={!newPrompt.trim()}
              >
                <PlusIcon className="h-4 w-4 mr-1" />
                Add Prompt
              </Button>
            </div>
          </div>
        )}

        {prompts.length >= maxPrompts && (
          <div className="text-center py-4 text-sm text-gray-500">
            Maximum of {maxPrompts} prompts reached
          </div>
        )}

        {/* Prompt tips and syntax guide */}
        <div className="bg-blue-50 rounded-lg p-4">
          <h4 className="text-sm font-medium text-blue-900 mb-2">Prompt Tips & Syntax</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h5 className="text-xs font-medium text-blue-800 mb-1">Best Practices</h5>
              <ul className="text-xs text-blue-800 space-y-1">
                <li>• Be specific and clear in your instructions</li>
                <li>• Include examples when possible</li>
                <li>• Test different phrasings to find what works best</li>
                <li>• Consider the context and desired output format</li>
              </ul>
            </div>
            <div>
              <h5 className="text-xs font-medium text-blue-800 mb-1">Syntax Highlighting</h5>
              <ul className="text-xs text-blue-800 space-y-1">
                <li>• <span className="text-blue-600 font-semibold">{'{variable}'}</span> - Variables</li>
                <li>• <span className="text-green-600 font-semibold">[instruction]</span> - Instructions</li>
                <li>• <span className="text-red-600 font-bold">System:</span> - Role indicators</li>
                <li>• <span className="font-bold">**bold**</span> and <span className="italic">*italic*</span> text</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default PromptInput;