import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { format, parseISO } from 'date-fns'; // Import date-fns functions

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Function to scroll smoothly to an element by its ID
export function scrollToElement(elementId: string) {
  const element = document.getElementById(elementId);
  if (element) {
    element.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
  }
}

// Function to format a date range
export function formatDateRange(startDateStr: string, endDateStr: string | null): string {
  const startDate = parseISO(startDateStr);
  const startFormatted = format(startDate, 'MMM yyyy');

  if (!endDateStr) {
    return `${startFormatted} – Present`;
  }

  const endDate = parseISO(endDateStr);
  // Check if end date is the same month and year as start date
  if (startDate.getFullYear() === endDate.getFullYear() && startDate.getMonth() === endDate.getMonth()) {
      return startFormatted; // If same month/year, just show start date
  }

  const endFormatted = format(endDate, 'MMM yyyy');
  return `${startFormatted} – ${endFormatted}`;
}
