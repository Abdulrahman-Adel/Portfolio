import { 
  Experience, 
  Project, 
  Skill, 
  Publication, 
  Certification, 
  SocialLink, 
  TechCategory,
  ProjectCategory
} from './types';

export const SOCIAL_LINKS: SocialLink[] = [
  {
    name: 'GitHub',
    url: 'https://github.com/Abdulrahman-Adel',
    icon: 'ri-github-fill'
  },
  {
    name: 'LinkedIn',
    url: 'https://www.linkedin.com/in/abdulrahman-adel',
    icon: 'ri-linkedin-box-fill'
  },
  {
    name: 'Kaggle',
    url: 'https://www.kaggle.com/',
    icon: 'ri-folder-chart-2-fill'
  },
  {
    name: 'Email',
    url: 'mailto:abdulrahman.adel098@gmail.com',
    icon: 'ri-mail-fill'
  }
];

export const EXPERIENCES: Experience[] = [
  {
    id: 1,
    title: 'Data Scientist',
    company: 'Value Driven Data',
    location: 'Dubai, United Arab Emirates',
    startDate: '2023-06-01',
    endDate: null,
    description: [
      'Drove data-driven solutions, transforming complex data into actionable insights.',
      'Developed AI-driven solutions including product matching using sentence transformers and brand extraction through NER models.',
      'Built two LLM-based chatbots on Snowflake native applications using Streamlit for natural language data interaction.',
      'Leveraged Power BI and Astrato to create data visualizations for real-time business insights.',
      'Collaborated with cross-functional teams to automate workflows and integrate ML solutions into the data analytics pipeline.'
    ],
    technologies: ['Machine Learning', 'NLP', 'Snowflake', 'Power BI', 'Streamlit']
  },
  {
    id: 2,
    title: 'Machine Learning Engineer',
    company: 'OMALINE',
    location: 'Riyadh, Saudi Arabia',
    startDate: '2023-01-01',
    endDate: '2023-06-01',
    description: [
      'Developed a dynamic search engine using Large Language Models (LLMs), significantly improving relevance.',
      'Engineered algorithms to optimize search result ranking, boosting user engagement.',
      'Orchestrated large-scale web scraping for data acquisition and model training.',
      'Fine-tuned LLMs for efficient processing of datasets.',
      'Built machine learning models for product categorization, enhancing search accuracy.'
    ],
    technologies: ['LLMs', 'Search Algorithms', 'Web Scraping', 'Model Fine-tuning']
  },
  {
    id: 3,
    title: 'Research And Development Engineer',
    company: 'Iskraemeco',
    location: 'Cairo, Egypt',
    startDate: '2021-10-01',
    endDate: '2022-06-01',
    description: [
      'Assisted in hardware design of Next Generation Smart-Grid Meter (NSGM).',
      'Developed C++ script for efficient data collection and transmission to cloud.',
      'Implemented PyTorch-based deep learning model for power consumption forecasting.',
      'Utilized variational autoencoder for non-intrusive load monitoring.',
      'Designed algorithm for harmonics and anomaly detection, enhancing grid stability.'
    ],
    technologies: ['PyTorch', 'C++', 'Time Series Analysis', 'Embedded Systems']
  },
  {
    id: 4,
    title: 'Junior Machine learning Engineer',
    company: 'aprcot',
    location: 'Cairo, Egypt',
    startDate: '2021-08-01',
    endDate: '2021-10-01',
    description: [
      'Developed Arabic Automatic Speech Recognition (ASR) prototype.',
      'Designed Transformer-based encoder-decoder architecture.',
      'Achieved 12% Word Error Rate (WER) on Mozilla Common Voice dataset.',
      'Utilized Google Cloud Platform (GCP) with GPU for efficient model training.',
      'Collaborated with development team for chatbot integration.'
    ],
    technologies: ['ASR', 'Transformer Models', 'GCP', 'Arabic NLP']
  },
  {
    id: 5,
    title: 'Machine Learning Researcher',
    company: 'WRL Wireless Research Lab',
    location: 'Cairo, Egypt',
    startDate: '2020-09-01',
    endDate: '2021-07-01',
    description: [
      'Developed AI-based module to detect cheating in online lab exams using mouse interaction analysis.',
      'Employed KNN, SVC, Random Forest, Logistic Regression, XGBoost, and LightGBM algorithms for classification.',
      'Conducted experiments validating effectiveness of approach with up to 90% accuracy using LightGBM.',
      'Achieved 88% precision and 95% degree of separation in cheat detection.'
    ],
    technologies: ['Classification Algorithms', 'XGBoost', 'LightGBM', 'Behavioral Analysis']
  }
];

export const PROJECT_CATEGORIES: ProjectCategory[] = [
  { id: 'all', name: 'All Projects' },
  { id: 'llm', name: 'LLM' },
  { id: 'ml', name: 'Machine Learning' },
  { id: 'dl', name: 'Deep Learning' },
  { id: 'nlp', name: 'NLP' },
  { id: 'cv', name: 'Computer Vision' },
  { id: 'open-source', name: 'Open Source' }
];

export const PROJECTS: Project[] = [
  {
    id: 0,
    title: 'PropertyGPT',
    description: 'An AI agent leveraging LLMs and chatbot technology to assist real estate professionals with lead generation, automated market analysis, and investment insights.',
    imageUrls: ['/assets/propertygpt.gif'],
    technologies: ['React', 'LLM', 'Chatbot', 'Full Stack'],
    projectUrl: '#',
    githubUrl: '#',
    category: 'llm'
  },
  {
    id: 1,
    title: 'Next Generation Smart-Grid Meter',
    description: 'Developed a deep learning model to forecast power consumption and implemented non-intrusive load monitoring using variational autoencoder.',
    imageUrls: [
      '/assets/nsgm/image96.png', 
      '/assets/nsgm/image28.jpg', 
      '/assets/nsgm/image105.png'
    ],
    technologies: ['PyTorch', 'Embedded Linux', 'C++'],
    projectUrl: '#',
    githubUrl: '#',
    category: 'dl'
  },
  {
    id: 2,
    title: 'Violence Detection in YouTube Videos',
    description: 'Coded vision transformer (ViT) architecture from scratch and trained on diverse YouTube dataset, achieving 85% accuracy in real-world scenarios.',
    imageUrls: ['https://images.unsplash.com/photo-1579167728798-a1cf3d595960?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=870&q=80'],
    technologies: ['TensorFlow', 'OpenCV', 'FastAPI'],
    projectUrl: '#',
    githubUrl: '#',
    category: 'cv'
  },
  {
    id: 3,
    title: 'OpenVINO Conjugate Transpose Operation',
    description: 'Implemented Conjugate Transpose operation in TensorFlow Frontend for Intel\'s OpenVINO and extended TF Frontend with corresponding loader.',
    imageUrls: ['https://images.unsplash.com/photo-1639322537228-f710d846310a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1332&q=80'],
    technologies: ['C++', 'Pytest', 'TensorFlow', 'OpenVINO'],
    projectUrl: '#',
    githubUrl: '#',
    category: 'open-source'
  },
  {
    id: 4,
    title: 'Arabic Speech Recognition System',
    description: 'Built an end-to-end Automatic Speech Recognition (ASR) system for Arabic language using Transformer architecture and achieved 12% WER.',
    imageUrls: ['https://images.unsplash.com/photo-1589254065878-42c9da997008?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=870&q=80'],
    technologies: ['Python', 'TensorFlow', 'NLP', 'Audio Processing'],
    projectUrl: '#',
    githubUrl: '#',
    category: 'nlp'
  },
  {
    id: 5,
    title: 'Cheating Detection in Online Exams',
    description: 'Developed a machine learning-based system to detect cheating in online examinations using mouse interaction patterns and achieved 90% accuracy.',
    imageUrls: ['https://images.unsplash.com/photo-1606326608606-aa0b62935f2b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=870&q=80'],
    technologies: ['Python', 'XGBoost', 'LightGBM', 'Scikit-learn'],
    projectUrl: '#',
    githubUrl: '#',
    category: 'ml'
  },
  {
    id: 6,
    title: 'LLM-based Product Search Engine',
    description: 'Developed a search engine using Large Language Models to improve relevance and user engagement for e-commerce product search.',
    imageUrls: ['https://images.unsplash.com/photo-1555952494-efd681c7e3f9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=870&q=80'],
    technologies: ['Python', 'PyTorch', 'LLMs', 'Vector Search'],
    projectUrl: '#',
    githubUrl: '#',
    category: 'nlp'
  }
];

export const TECH_CATEGORIES: TechCategory[] = [
  {
    title: 'Programming Languages',
    items: [
      { name: 'Python', icon: 'ri-python-fill', primary: true },
      { name: 'C++', icon: 'ri-code-box-fill' },
      { name: 'Java', icon: 'ri-code-box-fill' },
      { name: 'SQL', icon: 'ri-database-2-fill' }
    ]
  },
  {
    title: 'Machine Learning Frameworks',
    items: [
      { name: 'TensorFlow', primary: true },
      { name: 'PyTorch', primary: true },
      { name: 'Scikit-learn' },
      { name: 'XGBoost' },
      { name: 'LightGBM' }
    ]
  },
  {
    title: 'Data Science Libraries',
    items: [
      { name: 'Pandas' },
      { name: 'NumPy' },
      { name: 'Matplotlib' },
      { name: 'Seaborn' }
    ]
  },
  {
    title: 'BI & Data Warehousing',
    items: [
      { name: 'Snowflake' },
      { name: 'Power BI' },
      { name: 'Tableau' },
      { name: 'Astrato' }
    ]
  },
  {
    title: 'Cloud Platforms',
    items: [
      { name: 'AWS' },
      { name: 'Google Cloud' },
      { name: 'Azure' },
      { name: 'Docker' }
    ]
  }
];

export const SKILLS: Skill[] = [
  {
    name: 'Machine Learning',
    level: 95,
    proficiency: 'Expert'
  },
  {
    name: 'Deep Learning',
    level: 90,
    proficiency: 'Advanced'
  },
  {
    name: 'Natural Language Processing',
    level: 85,
    proficiency: 'Advanced'
  },
  {
    name: 'Computer Vision',
    level: 80,
    proficiency: 'Intermediate'
  },
  {
    name: 'Data Engineering',
    level: 75,
    proficiency: 'Intermediate'
  },
  {
    name: 'Web Development',
    level: 60,
    proficiency: 'Basic'
  }
];

export const PUBLICATIONS: Publication[] = [
  {
    id: 1,
    title: 'An Intelligent Approach for Fair Assessment of Online Laboratory Examinations in Laboratory Learning Systems Based on Student\'s Mouse Interaction Behavior',
    authors: 'Hassan Hosny, Hadeer A., Abdulrahman A. Ibrahim, Mahmoud M. Elmesalawy, and Ahmed M. Abd El-Haleem',
    journal: 'Applied Sciences, vol. 12, no. 22: 11416',
    year: '2022',
    description: 'This paper presents an intelligent approach for detecting cheating in online laboratory examinations by analyzing mouse interaction behavior, achieving up to 90% accuracy using LightGBM algorithms.',
    url: 'https://doi.org/10.3390/app122211416'
  },
  {
    id: 2,
    title: 'IoT Next Generation Smart Grid Meter (NGSM) for On-Edge Household Appliances Detection Based on Deep Learning and Embedded Linux',
    authors: 'N. E. -D. M. Mohamed, M. M. El-Dakroury. A. A. Ibrahim and G. A. Nfady',
    journal: '2023 5th Novel Intelligent and Leading Emerging Sciences Conference (NILES)',
    year: '2023',
    description: 'This paper presents the development of a Next Generation Smart Grid Meter that utilizes deep learning techniques for on-edge household appliance detection and monitoring, enhancing grid stability and efficiency.',
    url: 'https://doi.org/10.1109/NILES59815.2023.10296567'
  }
];

export const CERTIFICATIONS: Certification[] = [
  {
    id: 1,
    title: 'Computer Vision Nanodegree',
    issuer: 'Udacity',
    date: '2021',
    url: '#'
  },
  {
    id: 2,
    title: 'Deep Learning Specialization',
    issuer: 'Coursera',
    date: 'August 2020',
    url: '#'
  },
  {
    id: 3,
    title: 'Machine Learning Engineering for Production (MLOps)',
    issuer: 'Coursera',
    date: 'December 2023',
    url: '#'
  },
  {
    id: 4,
    title: 'Machine Learning with Python',
    issuer: 'IBM Badge',
    date: null,
    url: '#'
  }
];

export const EDUCATIONAL_BACKGROUND = [
  {
    degree: 'Bachelor of Engineering in Computer Engineering',
    school: 'Helwan University, Cairo',
    years: '2017 - 2022'
  },
  {
    degree: 'Computer Vision Nanodegree',
    school: 'Udacity',
    years: '2021'
  }
];

export const PERSONAL_INFO = {
  name: 'Abdulrahman Adel Ibrahim',
  title: 'Data Scientist & ML Engineer',
  location: 'Dubai, United Arab Emirates',
  currentRole: 'Data Scientist @ Value Driven Data',
  email: 'abdulrahman.adel098@gmail.com',
  phone: {
    primary: '(+971) 50 439 8923',
    secondary: '(+20) 1146631026'
  },
  languages: 'English, Arabic',
  shortBio: 'Passionate Data Scientist and Machine Learning Engineer with experience in developing impactful AI solutions across various industries.',
  longBio: `A highly motivated Data Scientist and Machine Learning Engineer dedicated to developing and implementing innovative AI solutions. With a proven track record, I excel at tackling complex challenges and transforming data into actionable business intelligence. My expertise spans key areas including Natural Language Processing (NLP), Large Language Models (LLMs), time series analysis, and computer vision.

Possessing strong technical skills in Python, C++, TensorFlow, PyTorch, and essential data science libraries, I am proficient across the full machine learning lifecycle – from data acquisition and preprocessing to model development, fine-tuning, and deployment. My experience extends to cloud platforms like AWS and GCP, and data warehousing solutions such as Snowflake. I have applied these capabilities to diverse applications, including building recommendation systems, developing chatbots, forecasting, contributing to open-source projects like OpenVINO, and implementing specialized models like Vision Transformers and Variational Autoencoders. I am passionate about continuous learning and applying cutting-edge technology to drive innovation and solve real-world problems.`,
  // Add other personal info fields as needed
};
