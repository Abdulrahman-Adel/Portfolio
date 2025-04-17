import { CERTIFICATIONS } from '@/lib/constants';
import { Card, CardContent } from '@/components/ui/card';
import { motion } from 'framer-motion';

export default function Certifications() {
  return (
    <section id="certifications" className="py-16 bg-gray-100">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <motion.h2 
          className="text-3xl font-bold mb-10 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          Certifications
        </motion.h2>
        
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {CERTIFICATIONS.map((certification, index) => (
              <motion.div 
                key={certification.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.1 * index }}
              >
                <Card className="bg-white rounded-lg shadow-sm h-full flex flex-col">
                  <CardContent className="p-6 flex flex-col flex-1">
                    <div className="flex-1">
                      <h3 className="font-semibold text-xl mb-2">{certification.title}</h3>
                      <p className="text-gray-700 mb-4">{certification.issuer}{certification.date ? `, ${certification.date}` : ''}</p>
                    </div>
                    <a 
                      href={certification.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center text-primary hover:text-blue-700 font-medium mt-auto"
                    >
                      View Certificate <i className="ri-external-link-line ml-1"></i>
                    </a>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
