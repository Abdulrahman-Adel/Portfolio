import { PUBLICATIONS } from '@/lib/constants';
import { Card, CardContent } from '@/components/ui/card';
import { motion } from 'framer-motion';

export default function Publications() {
  return (
    <section id="publications" className="py-16 bg-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <motion.h2 
          className="text-3xl font-bold mb-10 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          Publications
        </motion.h2>
        
        <div className="max-w-4xl mx-auto space-y-6">
          {PUBLICATIONS.map((publication, index) => (
            <motion.div 
              key={publication.id}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 * index }}
            >
              <Card className="bg-gray-100 rounded-lg shadow-sm">
                <CardContent className="p-6">
                  <h3 className="font-semibold text-xl mb-3">{publication.title}</h3>
                  <div className="flex flex-wrap items-center text-gray-700 mb-3 gap-x-2 gap-y-1">
                    <span className="font-medium">Authors:</span>
                    <span>{publication.authors}</span>
                  </div>
                  <div className="flex items-center text-gray-700 mb-3">
                    <span className="font-medium mr-2">Journal:</span>
                    <span>{publication.journal}</span>
                  </div>
                  <div className="flex items-center text-gray-700 mb-4">
                    <span className="font-medium mr-2">Year:</span>
                    <span>{publication.year}</span>
                  </div>
                  <p className="text-gray-700 mb-4">
                    {publication.description}
                  </p>
                  <a 
                    href={publication.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center text-primary hover:text-blue-700 font-medium"
                  >
                    Read Publication <i className="ri-external-link-line ml-1"></i>
                  </a>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
