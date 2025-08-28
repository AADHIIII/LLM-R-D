# Web Interface

This is the React-based web interface for the LLM Optimization Platform.

## Features

- **React 18** with TypeScript for type safety
- **React Router** for client-side routing
- **Tailwind CSS** for styling and responsive design
- **Heroicons** for consistent iconography
- **Axios** for API communication
- **Context API** for global state management
- **React Testing Library** for component testing

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Layout.tsx      # Main layout with sidebar navigation
│   ├── Button.tsx      # Button component with variants
│   ├── Card.tsx        # Card component for content sections
│   ├── Input.tsx       # Input component with validation
│   └── __tests__/      # Component tests
├── pages/              # Page components
│   ├── Dashboard.tsx   # Main dashboard
│   ├── FineTuning.tsx  # Fine-tuning interface
│   ├── PromptTesting.tsx # Prompt testing interface
│   ├── Results.tsx     # Results visualization
│   └── Analytics.tsx   # Analytics and cost tracking
├── services/           # API services
│   └── api.ts         # Axios-based API client
├── context/           # React Context providers
│   └── AppContext.tsx # Global application state
├── types/             # TypeScript type definitions
│   └── index.ts       # Common types
└── App.tsx            # Main application component
```

## Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App (not recommended)

## Development

1. Navigate to the frontend directory:
   ```bash
   cd web_interface/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open [http://localhost:3000](http://localhost:3000) to view in browser

## API Configuration

The frontend expects the API to be running on `http://localhost:5000/api/v1` by default. You can override this by setting the `REACT_APP_API_URL` environment variable.

Create a `.env.local` file in the frontend directory:
```
REACT_APP_API_URL=http://localhost:5000/api/v1
```

## Testing

Run tests with:
```bash
npm test
```

For coverage report:
```bash
npm test -- --coverage --watchAll=false
```

## Building for Production

```bash
npm run build
```

This creates a `build` folder with optimized production files.

## Next Steps

This is the foundation for the web interface. The following features will be implemented in subsequent tasks:

1. **Experiment Management Interface** (Task 7.2)
   - Dataset upload with progress tracking
   - Training configuration forms
   - Experiment dashboard with status monitoring

2. **Prompt Testing Interface** (Task 7.3)
   - Prompt input with syntax highlighting
   - Model selection dropdown
   - Side-by-side output comparison
   - Real-time evaluation results

3. **Analytics and Visualization** (Task 8)
   - Performance metrics charts
   - Cost tracking dashboard
   - Interactive data visualization