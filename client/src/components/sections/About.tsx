import { PERSONAL_INFO, EDUCATIONAL_BACKGROUND } from '@/lib/constants';
import { motion } from 'framer-motion';
import { Separator } from '@/components/ui/separator';

export default function About() {
  return (
    <section id="about" className="py-16 bg-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <motion.h2 
          className="text-3xl font-bold mb-10 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          About Me
        </motion.h2>
        
        <div className="max-w-4xl mx-auto">
          <motion.div 
            className="mb-8"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {PERSONAL_INFO.longBio.split('\n\n').map((paragraph, index) => (
              <p key={index} className="text-lg leading-relaxed mb-4">
                {paragraph}
              </p>
            ))}
          </motion.div>
          
          <motion.div 
            className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <div className="bg-gray-100 rounded-lg p-6">
              <h3 className="font-semibold text-xl mb-3 flex items-center">
                <i className="ri-map-pin-line text-primary mr-2"></i> Location
              </h3>
              <p>{PERSONAL_INFO.location}</p>
            </div>
            
            <div className="bg-gray-100 rounded-lg p-6">
              <h3 className="font-semibold text-xl mb-3 flex items-center">
                <i className="ri-group-line text-primary mr-2"></i> Current Role
              </h3>
              <p>{PERSONAL_INFO.currentRole}</p>
            </div>
            
            <div className="bg-gray-100 rounded-lg p-6">
              <h3 className="font-semibold text-xl mb-3 flex items-center">
                <i className="ri-mail-line text-primary mr-2"></i> Email
              </h3>
              <p>{PERSONAL_INFO.email}</p>
            </div>
            
            <div className="bg-gray-100 rounded-lg p-6">
              <h3 className="font-semibold text-xl mb-3 flex items-center">
                <i className="ri-translate-2 text-primary mr-2"></i> Languages
              </h3>
              <p>{PERSONAL_INFO.languages}</p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <h3 className="font-semibold text-xl mb-4">Education</h3>
            
            {EDUCATIONAL_BACKGROUND.map((education, index) => (
              <div key={index} className="bg-gray-100 rounded-lg p-6 mb-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-2">
                  <h4 className="font-medium text-lg">{education.degree}</h4>
                  <span className="text-gray-600 text-sm mt-1 sm:mt-0">{education.years}</span>
                </div>
                <p className="text-gray-700">{education.school}</p>
              </div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
