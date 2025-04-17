import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(date: Date | string) {
  return new Date(date).toLocaleDateString('en-US', {
    month: 'long',
    year: 'numeric',
  });
}

export function formatDateRange(startDate: string, endDate: string | null) {
  const start = formatDate(startDate);
  
  if (!endDate) {
    return `${start} - Present`;
  }
  
  const end = formatDate(endDate);
  return `${start} - ${end}`;
}

export function scrollToElement(id: string) {
  const element = document.getElementById(id);
  if (element) {
    element.scrollIntoView({ behavior: 'smooth' });
  }
}
