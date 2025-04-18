import { useState } from 'react';
import { PROJECTS, PROJECT_CATEGORIES } from '@/lib/constants';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { motion } from 'framer-motion';
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel"

export default function Projects() {
  const [activeCategory, setActiveCategory] = useState('all');

  const filteredProjects = activeCategory === 'all'
    ? PROJECTS
    : PROJECTS.filter(project => project.category === activeCategory);

  return (
    <section id="projects" className="py-16 bg-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <motion.h2 
          className="text-3xl font-bold mb-4 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          Projects
        </motion.h2>
        
        <motion.p 
          className="text-gray-700 text-center mb-10 max-w-2xl mx-auto"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          A collection of my significant technical projects in machine learning, deep learning, and data science.
        </motion.p>

        {/* Project Filters */}
        <motion.div 
          className="flex flex-wrap justify-center gap-3 mb-10"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          {PROJECT_CATEGORIES.map((category) => (
            <Button
              key={category.id}
              variant={activeCategory === category.id ? 'default' : 'outline'}
              onClick={() => setActiveCategory(category.id)}
              className="px-4 py-2 rounded-md text-sm font-medium"
            >
              {category.name}
            </Button>
          ))}
        </motion.div>

        {/* Projects Grid */}
        {/* TEST IMAGE REMOVED */}
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredProjects.map((project, index) => (
            <motion.div
              key={project.id}
              className={index === 0 ? "lg:col-span-2" : ""}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 * index }}
            >
              <Card className="flex flex-col bg-gray-100 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                <div className="h-48 flex-shrink-0 bg-gray-200 relative">
                  {project.imageUrls && project.imageUrls.length > 0 ? (
                    <img 
                      src={project.imageUrls[0]} 
                      alt={`${project.title} - Image 1`} 
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center bg-gray-300 text-gray-500">
                      No Image Available
                    </div>
                  )}
                  <div className="absolute top-3 right-3">
                    <Badge 
                      variant={
                        project.category === 'dl' ? 'green' :
                        project.category === 'cv' ? 'purple' :
                        project.category === 'nlp' ? 'yellow' :
                        project.category === 'open-source' ? 'blue' :
                        'gray'
                      }
                      className="bg-opacity-90 text-white rounded"
                    >
                      {PROJECT_CATEGORIES.find(c => c.id === project.category)?.name.replace('All ', '')}
                    </Badge>
                  </div>
                </div>
                
                <CardContent className="p-6 flex-grow">
                  <h3 className="font-semibold text-xl mb-2">{project.title}</h3>
                  <p className="text-gray-700 mb-4">{project.description}</p>
                  
                  <div className="flex flex-wrap gap-2 mb-4">
                    {project.technologies.map((tech, idx) => (
                      <span key={idx} className="inline-block px-2 py-1 text-xs font-medium bg-gray-200 text-gray-800 rounded">
                        {tech}
                      </span>
                    ))}
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <a 
                      href={project.projectUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:text-blue-700 font-medium flex items-center"
                    >
                      View Project <i className="ri-external-link-line ml-1"></i>
                    </a>
                    <a 
                      href={project.githubUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-gray-700 hover:text-gray-900"
                    >
                      <i className="ri-github-fill text-lg"></i>
                    </a>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
        
        {PROJECTS.length > 6 && (
          <motion.div 
            className="text-center mt-10"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Button variant="primaryOutline" size="lg">
              View All Projects <i className="ri-arrow-right-line ml-2"></i>
            </Button>
          </motion.div>
        )}
      </div>
    </section>
  );
}
