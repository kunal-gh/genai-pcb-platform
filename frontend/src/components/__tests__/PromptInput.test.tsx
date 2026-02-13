import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import PromptInput from '../PromptInput';

describe('PromptInput', () => {
  it('renders prompt input component', () => {
    render(<PromptInput onSubmit={jest.fn()} />);
    expect(screen.getByText('Describe Your PCB Design')).toBeInTheDocument();
  });

  it('shows error for short prompts', () => {
    render(<PromptInput onSubmit={jest.fn()} />);
    const input = screen.getByPlaceholderText(/Example:/);
    const button = screen.getByText('Generate Design');
    
    fireEvent.change(input, { target: { value: 'short' } });
    fireEvent.click(button);
    
    expect(screen.getByText('Prompt must be at least 10 characters long')).toBeInTheDocument();
  });

  it('calls onSubmit with valid prompt', () => {
    const onSubmit = jest.fn();
    render(<PromptInput onSubmit={onSubmit} />);
    
    const input = screen.getByPlaceholderText(/Example:/);
    const validPrompt = 'Create a simple LED circuit with resistor';
    
    fireEvent.change(input, { target: { value: validPrompt } });
    fireEvent.click(screen.getByText('Generate Design'));
    
    expect(onSubmit).toHaveBeenCalledWith(validPrompt);
  });

  it('disables input when disabled prop is true', () => {
    render(<PromptInput onSubmit={jest.fn()} disabled={true} />);
    const input = screen.getByPlaceholderText(/Example:/);
    expect(input).toBeDisabled();
  });
});
