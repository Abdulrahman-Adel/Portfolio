import { EXPERIENCES } from '@/lib/constants';
import { Badge } from '@/components/ui/badge';
import { formatDateRange } from '@/lib/utils';
import { motion } from 'framer-motion';

export default function Experience() {
  return (
    <section id="experience" className="py-16 bg-gray-100">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <motion.h2 
          className="text-3xl font-bold mb-10 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          Work Experience
        </motion.h2>
        
        <div className="max-w-4xl mx-auto">
          <div className="relative">
            {EXPERIENCES.map((experience, index) => (
              <motion.div 
                key={experience.id}
                className="timeline-item relative pl-10 pb-12"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <div className="timeline-dot absolute left-0 top-0">
                  <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center relative">
                    <i className="ri-briefcase-line text-white"></i>
                    
                    {/* Vertical line */}
                    {index < EXPERIENCES.length - 1 && (
                      <div className="absolute left-1/2 transform -translate-x-1/2 top-full h-full w-[2px] bg-gray-200"></div>
                    )}
                  </div>
                </div>
                
                <div className="bg-white rounded-lg shadow-sm p-6">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3">
                    <h3 className="font-semibold text-xl">{experience.title}</h3>
                    <Badge 
                      variant="blue" 
                      className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium mt-2 sm:mt-0"
                    >
                      {formatDateRange(experience.startDate, experience.endDate)}
                    </Badge>
                  </div>
                  
                  <div className="mb-3">
                    <span className="font-medium">{experience.company}</span>
                    <span className="text-gray-700"> • {experience.location}</span>
                  </div>
                  
                  <ul className="list-disc pl-5 space-y-2 text-gray-700">
                    {experience.description.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                  
                  <div className="mt-4 flex flex-wrap gap-2">
                    {experience.technologies.map((tech, idx) => (
                      <span key={idx} className="inline-block px-2 py-1 text-xs font-medium bg-gray-100 text-gray-800 rounded">
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
              </motion.div>
            ))}
            
            {EXPERIENCES.length > 5 && (
              <div className="pl-10">
                <a href="#" className="inline-flex items-center text-primary hover:text-blue-700 font-medium">
                  View more experience
                  <i className="ri-arrow-right-line ml-1"></i>
                </a>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
