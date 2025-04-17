import { PERSONAL_INFO, SOCIAL_LINKS } from '@/lib/constants';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { scrollToElement } from '@/lib/utils';
import { motion } from 'framer-motion';

export default function Hero() {
  return (
    <section 
      id="hero" 
      className="pt-28 pb-20 md:pt-32 md:pb-24 bg-gradient-to-br from-white to-gray-100"
    >
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="flex flex-col-reverse md:flex-row items-center md:items-start gap-10">
            <motion.div 
              className="w-full md:w-2/3"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h1 className="text-4xl md:text-5xl font-bold mb-4">
                <span className="block">Hi, I'm {PERSONAL_INFO.name.split(' ')[0]}</span>
                <span className="text-primary block mt-2">{PERSONAL_INFO.title}</span>
              </h1>
              <p className="text-lg text-gray-700 mb-6">{PERSONAL_INFO.shortBio}</p>
              
              <div className="flex flex-wrap gap-3 mb-8">
                <Badge variant="blue" className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium">
                  <i className="ri-database-2-line mr-1.5"></i> Data Analysis
                </Badge>
                <Badge variant="green" className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium">
                  <i className="ri-code-box-line mr-1.5"></i> Machine Learning
                </Badge>
                <Badge variant="purple" className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium">
                  <i className="ri-brain-line mr-1.5"></i> Deep Learning
                </Badge>
                <Badge variant="yellow" className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium">
                  <i className="ri-chat-3-line mr-1.5"></i> NLP
                </Badge>
              </div>
              
              <div className="flex flex-wrap gap-4">
                <Button 
                  size="hero"
                  onClick={() => scrollToElement('contact')}
                >
                  Get in Touch
                </Button>
                <Button 
                  variant="primaryOutline" 
                  size="hero"
                  onClick={() => scrollToElement('projects')}
                >
                  View My Work
                </Button>
              </div>
            </motion.div>
            
            <motion.div 
              className="w-40 h-40 md:w-48 md:h-48 rounded-full overflow-hidden bg-gradient-to-r from-primary to-purple-500 shadow-lg"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <div className="w-full h-full bg-gradient-to-r from-primary to-purple-500 flex items-center justify-center text-white text-4xl font-bold">
                AA
              </div>
            </motion.div>
          </div>
          
          <motion.div 
            className="mt-12 flex flex-wrap justify-center md:justify-start gap-5"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            {SOCIAL_LINKS.map((link) => (
              <a 
                key={link.name}
                href={link.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-700 hover:text-primary transition-colors"
                aria-label={link.name}
              >
                <i className={`${link.icon} text-2xl`}></i>
              </a>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
