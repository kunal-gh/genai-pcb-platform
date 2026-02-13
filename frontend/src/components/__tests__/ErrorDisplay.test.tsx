import React from 'react';
import { render, screen } from '@testing-library/react';
import ErrorDisplay, { DesignError } from '../ErrorDisplay';

describe('ErrorDisplay', () => {
  const mockErrors: DesignError[] = [
    {
      type: 'error',
      category: 'ERC',
      message: 'Unconnected pin detected',
      suggestion: 'Connect pin to appropriate net',
      location: 'U1 pin 5',
    },
    {
      type: 'warning',
      category: 'DRC',
      message: 'Trace width below minimum',
      suggestion: 'Increase trace width to 0.2mm',
    },
  ];

  it('renders errors grouped by category', () => {
    render(<ErrorDisplay errors={mockErrors} />);
    expect(screen.getByText(/ERC/)).toBeInTheDocument();
    expect(screen.getByText(/DRC/)).toBeInTheDocument();
  });

  it('displays error details', () => {
    render(<ErrorDisplay errors={mockErrors} />);
    expect(screen.getByText('Unconnected pin detected')).toBeInTheDocument();
    expect(screen.getByText(/Connect pin to appropriate net/)).toBeInTheDocument();
    expect(screen.getByText(/U1 pin 5/)).toBeInTheDocument();
  });

  it('renders nothing when no errors', () => {
    const { container } = render(<ErrorDisplay errors={[]} />);
    expect(container.firstChild).toBeNull();
  });
});
