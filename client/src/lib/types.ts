export interface SocialLink {
  name: string;
  url: string;
  icon: string;
}

export interface Experience {
  id: number;
  title: string;
  company: string;
  location: string;
  startDate: string;
  endDate: string | null;
  description: string[];
  technologies: string[];
}

export interface ProjectCategory {
  id: string;
  name: string;
}

export interface Project {
  id: number;
  title: string;
  description: string;
  imageUrls: string[];
  technologies: string[];
  projectUrl: string;
  githubUrl: string;
  category: string;
}

export interface TechItem {
  name: string;
  icon?: string;
  primary?: boolean;
}

export interface TechCategory {
  title: string;
  items: TechItem[];
}

export interface Skill {
  name: string;
  level: number;
  proficiency: string;
}

export interface Publication {
  id: number;
  title: string;
  authors: string;
  journal: string;
  year: string;
  description: string;
  url: string;
}

export interface Certification {
  id: number;
  title: string;
  issuer: string;
  date: string | null;
  url: string;
}

export interface ContactFormData {
  name: string;
  email: string;
  subject: string;
  message: string;
}
