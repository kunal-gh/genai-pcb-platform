import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import SchematicPreview from '../SchematicPreview';

describe('SchematicPreview', () => {
  it('renders preview with image', () => {
    render(<SchematicPreview imageUrl="http://example.com/image.png" title="Test Preview" />);
    expect(screen.getByText('Test Preview')).toBeInTheDocument();
    expect(screen.getByAltText('Test Preview')).toBeInTheDocument();
  });

  it('shows placeholder when no image', () => {
    render(<SchematicPreview title="Test Preview" />);
    expect(screen.getByText('Preview not available')).toBeInTheDocument();
  });

  it('handles zoom controls', () => {
    render(<SchematicPreview imageUrl="http://example.com/image.png" title="Test Preview" />);
    
    const zoomInButton = screen.getAllByRole('button')[1];
    const zoomOutButton = screen.getAllByRole('button')[0];
    
    expect(screen.getByText('100%')).toBeInTheDocument();
    
    fireEvent.click(zoomInButton);
    expect(screen.getByText('125%')).toBeInTheDocument();
    
    fireEvent.click(zoomOutButton);
    expect(screen.getByText('100%')).toBeInTheDocument();
  });
});
