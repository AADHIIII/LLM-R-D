import React from 'react';

const Results: React.FC = () => {
  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Results</h1>
        <p className="mt-2 text-gray-600">
          View and analyze experiment results
        </p>
      </div>

      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900">
            Results Dashboard
          </h3>
          <div className="mt-5">
            <div className="text-center py-12">
              <p className="text-gray-500">Results dashboard coming soon</p>
              <p className="text-sm text-gray-400 mt-2">
                This will include comparative analysis and visualization
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;