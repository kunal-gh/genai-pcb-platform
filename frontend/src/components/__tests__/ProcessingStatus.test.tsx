import React from 'react';
import { render, screen } from '@testing-library/react';
import ProcessingStatus from '../ProcessingStatus';

describe('ProcessingStatus', () => {
  it('shows processing status with progress', () => {
    render(
      <ProcessingStatus
        status="processing"
        progress={50}
        currentStep="Generating SKiDL code"
      />
    );
    
    expect(screen.getByText('Processing Status')).toBeInTheDocument();
    expect(screen.getByText('Generating SKiDL code')).toBeInTheDocument();
    expect(screen.getByText('50% complete')).toBeInTheDocument();
  });

  it('shows success message when completed', () => {
    render(
      <ProcessingStatus
        status="completed"
        progress={100}
        currentStep="Completed"
      />
    );
    
    expect(screen.getByText('Design completed successfully!')).toBeInTheDocument();
  });

  it('shows error message when failed', () => {
    render(
      <ProcessingStatus
        status="failed"
        progress={30}
        currentStep="Failed"
        errorMessage="Invalid component specification"
      />
    );
    
    expect(screen.getByText('Invalid component specification')).toBeInTheDocument();
  });
});
